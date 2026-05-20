import csv
import json
import os
import tempfile
from collections import defaultdict

import pytest

from tipp3.configs import Configs
from tipp3.query_abundance import (
    RANKS,
    getAbundanceProfile,
    getSpeciesDetection,
    parseTaxonomy,
    parseSpeciesToMarker,
    reorderJson,
    updateAbundanceProfile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_classification(path, rows):
    """Write a CSV classification file (no header)."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def write_taxonomy(path, rows):
    """Write a taxonomy CSV with header: taxid,parent_id,rank,taxname."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['taxid', 'parent_id', 'rank', 'taxname'])
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# updateAbundanceProfile
# ---------------------------------------------------------------------------

class TestUpdateAbundanceProfile:
    def _make_profile(self):
        return {rank: defaultdict(float) for rank in RANKS}

    def test_accumulates_support_values(self, tmp_path):
        clas = tmp_path / "cls.csv"
        write_classification(clas, [
            ['read1', '9606', 'Homo sapiens', 'species', '0.8'],
            ['read2', '9606', 'Homo sapiens', 'species', '0.6'],
        ])
        profile = self._make_profile()
        updateAbundanceProfile(str(clas), profile)
        assert profile['species'][9606] == pytest.approx(1.4)

    def test_skips_short_lines(self, tmp_path):
        clas = tmp_path / "cls.csv"
        write_classification(clas, [
            ['read1', '9606', 'Homo sapiens'],          # only 3 fields
            ['read2', '9606', 'Homo sapiens', 'species', '0.5'],
        ])
        profile = self._make_profile()
        updateAbundanceProfile(str(clas), profile)
        assert profile['species'][9606] == pytest.approx(0.5)

    def test_skips_non_numeric_taxid(self, tmp_path):
        clas = tmp_path / "cls.csv"
        write_classification(clas, [
            ['read1', 'NA', 'unknown', 'species', '0.9'],
            ['read2', '1234', 'Bacteria', 'phylum', '0.7'],
        ])
        profile = self._make_profile()
        updateAbundanceProfile(str(clas), profile)
        assert profile['species'] == {}
        assert profile['phylum'][1234] == pytest.approx(0.7)

    def test_skips_non_numeric_support(self, tmp_path):
        clas = tmp_path / "cls.csv"
        write_classification(clas, [
            ['read1', '9606', 'Homo sapiens', 'species', 'NA'],
        ])
        profile = self._make_profile()
        updateAbundanceProfile(str(clas), profile)
        assert profile['species'] == {}

    def test_ignores_unknown_rank(self, tmp_path):
        clas = tmp_path / "cls.csv"
        write_classification(clas, [
            ['read1', '9606', 'Homo sapiens', 'strain', '0.9'],
        ])
        profile = self._make_profile()
        updateAbundanceProfile(str(clas), profile)
        for rank in RANKS:
            assert profile[rank] == {}

    def test_multiple_ranks_accumulated(self, tmp_path):
        clas = tmp_path / "cls.csv"
        write_classification(clas, [
            ['read1', '9606', 'Homo sapiens', 'species', '0.8'],
            ['read1', '9605', 'Homo', 'genus', '0.9'],
            ['read2', '9606', 'Homo sapiens', 'species', '0.3'],
        ])
        profile = self._make_profile()
        updateAbundanceProfile(str(clas), profile)
        assert profile['species'][9606] == pytest.approx(1.1)
        assert profile['genus'][9605] == pytest.approx(0.9)

    def test_empty_file_leaves_profile_unchanged(self, tmp_path):
        clas = tmp_path / "empty.csv"
        clas.write_text('')
        profile = self._make_profile()
        updateAbundanceProfile(str(clas), profile)
        for rank in RANKS:
            assert profile[rank] == {}


# ---------------------------------------------------------------------------
# getAbundanceProfile
# ---------------------------------------------------------------------------

class TestGetAbundanceProfile:
    def _make_taxonomy(self, tmp_path):
        tax_path = tmp_path / "taxonomy.csv"
        write_taxonomy(tax_path, [
            [9606, 9605, 'species', 'Homo sapiens'],
            [9605, 9604, 'genus', 'Homo'],
        ])
        return tax_path

    def test_normalizes_abundances_to_one(self, tmp_path):
        clas = tmp_path / "cls.csv"
        write_classification(clas, [
            ['read1', '9606', 'Homo sapiens', 'species', '2.0'],
            ['read2', '1234', 'Bacteria', 'species', '3.0'],
        ])
        tax_path = self._make_taxonomy(tmp_path)
        Configs.outdir = str(tmp_path)

        refpkg = {'taxonomy': {'taxonomy': str(tax_path)}}
        getAbundanceProfile(refpkg, {'marker1': str(clas)})

        outfile = tmp_path / 'abundance.species.tsv'
        assert outfile.exists()
        with open(outfile) as f:
            lines = f.read().strip().splitlines()
        # header + 2 data rows
        assert lines[0] == 'taxa\ttaxid\tabundance'
        abundances = [float(l.split('\t')[2]) for l in lines[1:]]
        assert sum(abundances) == pytest.approx(1.0)

    def test_outputs_file_per_rank(self, tmp_path):
        clas = tmp_path / "cls.csv"
        write_classification(clas, [
            ['read1', '9606', 'Homo sapiens', 'species', '1.0'],
            ['read1', '9605', 'Homo', 'genus', '1.0'],
        ])
        tax_path = self._make_taxonomy(tmp_path)
        Configs.outdir = str(tmp_path)

        refpkg = {'taxonomy': {'taxonomy': str(tax_path)}}
        getAbundanceProfile(refpkg, {'marker1': str(clas)})

        assert (tmp_path / 'abundance.species.tsv').exists()
        assert (tmp_path / 'abundance.genus.tsv').exists()

    def test_skips_rank_with_no_data(self, tmp_path):
        clas = tmp_path / "cls.csv"
        write_classification(clas, [
            ['read1', '9606', 'Homo sapiens', 'species', '1.0'],
        ])
        tax_path = self._make_taxonomy(tmp_path)
        Configs.outdir = str(tmp_path)

        refpkg = {'taxonomy': {'taxonomy': str(tax_path)}}
        getAbundanceProfile(refpkg, {'marker1': str(clas)})

        # genus had no data — file should not be written
        assert not (tmp_path / 'abundance.genus.tsv').exists()

    def test_unclassified_taxid_zero(self, tmp_path):
        clas = tmp_path / "cls.csv"
        write_classification(clas, [
            ['read1', '0', 'unclassified', 'species', '1.0'],
        ])
        tax_path = self._make_taxonomy(tmp_path)
        Configs.outdir = str(tmp_path)

        refpkg = {'taxonomy': {'taxonomy': str(tax_path)}}
        getAbundanceProfile(refpkg, {'marker1': str(clas)})

        outfile = tmp_path / 'abundance.species.tsv'
        content = outfile.read_text()
        assert 'unclassified\t0' in content

    def test_aggregates_across_multiple_markers(self, tmp_path):
        clas1 = tmp_path / "cls1.csv"
        clas2 = tmp_path / "cls2.csv"
        write_classification(clas1, [
            ['read1', '9606', 'Homo sapiens', 'species', '1.0'],
        ])
        write_classification(clas2, [
            ['read2', '9606', 'Homo sapiens', 'species', '3.0'],
        ])
        tax_path = self._make_taxonomy(tmp_path)
        Configs.outdir = str(tmp_path)

        refpkg = {'taxonomy': {'taxonomy': str(tax_path)}}
        getAbundanceProfile(refpkg, {'m1': str(clas1), 'm2': str(clas2)})

        outfile = tmp_path / 'abundance.species.tsv'
        lines = outfile.read_text().strip().splitlines()
        row = [l for l in lines[1:] if l.startswith('Homo sapiens')][0]
        abundance = float(row.split('\t')[2])
        assert abundance == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# parseTaxonomy
# ---------------------------------------------------------------------------

class TestParseTaxonomy:
    def test_basic_parsing(self, tmp_path):
        tax = tmp_path / "taxonomy.csv"
        write_taxonomy(tax, [
            [9606, 9605, 'species', 'Homo sapiens'],
            [9605, 9604, 'genus', 'Homo'],
        ])
        result = parseTaxonomy(str(tax))
        assert result[9606] == ('Homo sapiens', 9605, 'species')
        assert result[9605] == ('Homo', 9604, 'genus')

    def test_skips_header(self, tmp_path):
        tax = tmp_path / "taxonomy.csv"
        write_taxonomy(tax, [[1, 1, 'root', 'root']])
        result = parseTaxonomy(str(tax))
        assert 1 in result  # only the data row, header excluded

    def test_skips_short_rows(self, tmp_path):
        tax = tmp_path / "taxonomy.csv"
        with open(tax, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['taxid', 'parent_id', 'rank', 'taxname'])
            writer.writerow([9606, 9605, 'species'])   # only 3 data fields
        result = parseTaxonomy(str(tax))
        assert 9606 not in result

    def test_handles_comma_in_taxname(self, tmp_path):
        tax = tmp_path / "taxonomy.csv"
        with open(tax, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['taxid', 'parent_id', 'rank', 'taxname'])
            writer.writerow([9606, 9605, 'species', 'Homo, sapiens variant'])
        result = parseTaxonomy(str(tax))
        assert result[9606][0] == 'Homo, sapiens variant'


# ---------------------------------------------------------------------------
# parseSpeciesToMarker
# ---------------------------------------------------------------------------

class TestParseSpeciesToMarker:
    def _write_s2m(self, path, rows):
        with open(path, 'w') as f:
            f.write('tax_id\tname\tmarkers\n')
            for row in rows:
                f.write('\t'.join(str(x) for x in row) + '\n')

    def test_basic_parsing(self, tmp_path):
        s2m = tmp_path / "s2m.tsv"
        self._write_s2m(s2m, [[9606, 'Homo sapiens', 'markerA,markerB']])
        result = parseSpeciesToMarker(str(s2m))
        assert result[9606] == ['markerA', 'markerB']

    def test_skips_header(self, tmp_path):
        s2m = tmp_path / "s2m.tsv"
        self._write_s2m(s2m, [[9606, 'Homo sapiens', 'markerA']])
        result = parseSpeciesToMarker(str(s2m))
        assert 'tax_id' not in str(result)

    def test_skips_short_lines(self, tmp_path):
        s2m = tmp_path / "s2m.tsv"
        with open(s2m, 'w') as f:
            f.write('tax_id\tname\n')   # only 2 columns
            f.write('9606\tHomo sapiens\n')
        result = parseSpeciesToMarker(str(s2m))
        assert result == {}

    def test_multiple_species(self, tmp_path):
        s2m = tmp_path / "s2m.tsv"
        self._write_s2m(s2m, [
            [9606, 'Homo sapiens', 'markerA'],
            [9913, 'Bos taurus', 'markerB,markerC'],
        ])
        result = parseSpeciesToMarker(str(s2m))
        assert result[9606] == ['markerA']
        assert result[9913] == ['markerB', 'markerC']


# ---------------------------------------------------------------------------
# reorderJson
# ---------------------------------------------------------------------------

class TestReorderJson:
    EXPECTED_FIELDS = ['edge_num', 'likelihood', 'like_weight_ratio',
                       'distal_length', 'pendant_length']

    def _make_jplace(self, fields, placements):
        return {
            'fields': fields,
            'tree': '(a,b);',
            'placements': placements,
            'metadata': {'invocation': 'test'},
            'version': 3,
        }

    def test_fields_reordered(self, tmp_path):
        src = tmp_path / "in.jplace"
        dst = tmp_path / "out.jplace"
        fields = ['likelihood', 'edge_num', 'like_weight_ratio',
                  'distal_length', 'pendant_length']
        # one placement with 'n' format
        placements = [{'p': [[0.1, 42, 0.9, 0.01, 0.05]], 'n': ['read1']}]
        with open(src, 'w') as f:
            json.dump(self._make_jplace(fields, placements), f)

        reorderJson(str(src), str(dst))

        with open(dst) as f:
            obj = json.load(f)
        assert obj['fields'] == self.EXPECTED_FIELDS

    def test_placement_values_remapped(self, tmp_path):
        src = tmp_path / "in.jplace"
        dst = tmp_path / "out.jplace"
        fields = ['likelihood', 'edge_num', 'like_weight_ratio',
                  'distal_length', 'pendant_length']
        placements = [{'p': [[0.1, 42, 0.9, 0.01, 0.05]], 'n': ['read1']}]
        with open(src, 'w') as f:
            json.dump(self._make_jplace(fields, placements), f)

        reorderJson(str(src), str(dst))

        with open(dst) as f:
            obj = json.load(f)
        p = obj['placements'][0]['p'][0]
        # expected order: edge_num=42, likelihood=0.1, like_weight_ratio=0.9,
        #                 distal_length=0.01, pendant_length=0.05
        assert p == [42, 0.1, 0.9, 0.01, 0.05]

    def test_nm_format_flattened(self, tmp_path):
        src = tmp_path / "in.jplace"
        dst = tmp_path / "out.jplace"
        fields = ['edge_num', 'likelihood', 'like_weight_ratio',
                  'distal_length', 'pendant_length']
        # 'nm' format: each nm entry is [name, multiplicity]
        placements = [{'p': [[7, 0.2, 0.8, 0.02, 0.03]],
                       'nm': [['read1', 1], ['read2', 2]]}]
        with open(src, 'w') as f:
            json.dump(self._make_jplace(fields, placements), f)

        reorderJson(str(src), str(dst))

        with open(dst) as f:
            obj = json.load(f)
        # 'nm' names extracted; placement written under 'n'
        assert obj['placements'][0]['n'] == ['read1', 'read2']

    def test_metadata_and_version_preserved(self, tmp_path):
        src = tmp_path / "in.jplace"
        dst = tmp_path / "out.jplace"
        fields = ['edge_num', 'likelihood', 'like_weight_ratio',
                  'distal_length', 'pendant_length']
        placements = [{'p': [[1, 0.5, 0.5, 0.1, 0.1]], 'n': ['r1']}]
        jplace = self._make_jplace(fields, placements)
        jplace['metadata'] = {'invocation': 'my-cmd'}
        jplace['version'] = 3
        with open(src, 'w') as f:
            json.dump(jplace, f)

        reorderJson(str(src), str(dst))

        with open(dst) as f:
            obj = json.load(f)
        assert obj['metadata'] == {'invocation': 'my-cmd'}
        assert obj['version'] == 3
