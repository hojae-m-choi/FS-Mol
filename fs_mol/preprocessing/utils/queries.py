""" Queries and fieldnames used to extract all data from ChEMBL """

CHEMBL_ASSAY_PROTEIN = (
    "SELECT s.canonical_smiles AS smiles, act.pchembl_value AS pchembl,"
    " act.standard_value AS standard_value,"
    " act.standard_units AS standard_units,"
    " act.standard_relation AS standard_relation,"
    " act.standard_type AS activity_type,"
    " act.activity_comment AS activity_comment,"
    " a.chembl_id AS chembl_id,"
    " a.assay_type AS assay_type,"
    " a.assay_organism AS organism,"
    " a.confidence_score AS confidence_score,"
    " td.tid AS target_id,"
    " td.pref_name AS target,"
    " tt.target_type AS target_type,"
    " protcls.protein_class_id AS protein_id,"
    " protcls.pref_name AS protein_class_name,"
    " protcls.short_name AS protein_short_name,"
    " protcls.class_level AS protein_class_level,"
    " protcls.protein_class_desc AS protein_class_desc"
    " FROM assays a"
    " JOIN activities act ON a.assay_id = act.assay_id"
    " JOIN compound_structures s ON act.molregno = s.molregno"
    " JOIN target_dictionary td on td.tid = a.tid"
    " JOIN target_components tc on td.tid = tc.tid"
    " JOIN target_type tt on tt.target_type = td.target_type"
    " JOIN component_class compcls on tc.component_id = compcls.component_id"
    " JOIN protein_classification protcls on protcls.protein_class_id = compcls.protein_class_id"
    " AND a.chembl_id = {}"
)

DISTINCT_TABLES = {
    "activity_type": ("SELECT DISTINCT d.chembl_id, d.activity_type FROM ({}) AS d;"),
    "activity_comment": ("SELECT DISTINCT d.chembl_id, d.activity_comment FROM ({}) AS d;"),
    "standard_unit": ("SELECT DISTINCT d.chembl_id, d.standard_units FROM ({}) AS d;"),
    "target_id": ("SELECT DISTINCT  d.chembl_id, d.target_id FROM ({}) AS d;"),
    "protein_class_level": (
        " SELECT DISTINCT d.chembl_id, d.protein_class_level AS protein_class_level"
        " FROM ({}) AS d;"
    ),
    "target_type": (
        " SELECT DISTINCT d.chembl_id, d.target_type AS target_type" " FROM ({}) AS d;"
    ),
}

EXTENDED_SINGLE_ASSAY_NOPROTEIN = (
    "SELECT s.canonical_smiles AS smiles,"
    " act.pchembl_value AS pchembl,"
    " act.standard_value AS value,"
    " act.standard_units AS units,"
    " act.standard_relation AS relation,"
    " act.standard_type AS activity_type,"
    " act.activity_comment AS comment,"
    " a.chembl_id AS chembl_id,"
    " a.assay_type AS assay_type,"
    " a.assay_organism AS organism,"
    " a.confidence_score AS confidence_score,"
    " a.assay_cell_type AS cell_type,"
    " a.assay_tissue AS tissue"
    " FROM assays a"
    " JOIN activities act on a.assay_id = act.assay_id"
    " JOIN compound_structures s"
    " ON act.molregno = s.molregno AND a.chembl_id = {}"
)

COUNT_QUERIES = {
    "num_activity_type": "SELECT count(e.activity_type) AS num_activity_type FROM ({}) AS e GROUP BY e.chembl_id;",
    "num_activity_comment": "SELECT count(e.activity_comment) AS num_activity_comment FROM ({}) AS e GROUP BY e.chembl_id;",
    "num_standard_unit": "SELECT count(e.standard_units) AS num_standard_unit FROM ({}) AS e GROUP BY e.chembl_id;",
    "num_target_id": "SELECT count(e.target_id) AS num_target_id FROM ({}) AS e GROUP BY e.chembl_id;",
    "num_protein_class_level": "SELECT count(e.protein_class_level) AS num_protein_class_level FROM ({}) AS e GROUP BY e.chembl_id;",
    "num_target_type": "SELECT count(e.target_type) AS num_target_type FROM ({}) AS e GROUP BY e.chembl_id;",
}

FIELDNAMES = [
    "smiles",
    "pchembl",
    "standard_value",
    "standard_units",
    "standard_relation",
    "activity_type",
    "activity_comment",
    "chembl_id",
    "assay_type",
    "assay_organism",
    "confidence_score",
]

PROTEIN_FIELDS = [
    "target_id",
    "target",
    "target_type",
    "protein_id",
    "protein_class_name",
    "protein_short_name",
    "protein_class_level",
    "protein_class_desc",
]

CELL_FIELDS = [
    "assay_cell_type",
    "assay_tissue",
]

SUMMARY_FIELDNAMES = [
    "activity_type",
    "activity_comment",
    "standard_unit",
    "target_id",
    "protein_class_level",
    "target_type",
]
COUNTED_SUMMARY_FIELDNAMES = [
    "chembl_id",
    "num_activity_type",
    "num_activity_comment",
    "num_standard_unit",
    "num_target_id",
    "num_protein_class_level",
    "num_target_type",
    "size",
]
