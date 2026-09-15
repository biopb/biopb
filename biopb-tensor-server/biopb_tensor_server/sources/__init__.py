"""Source lifecycle: config-entry resolution (``resolve``), filesystem scan
orchestration (``source_manager`` + ``tree_scanner``) and confirmed-catalog
reconciliation (``reconciler``). Depends on ``core`` and ``adapters``; the
``serving`` server and its ``metadata_db`` are injected, named here only in type
annotations.
"""
