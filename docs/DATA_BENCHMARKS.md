# Data Benchmarks

Built from the local official-source pipeline.

| Metric | Measured Value |
|---|---:|
| CFPB records processed | 250,000 |
| Date coverage | 2016-03-02 to 2026-08-09 |
| Public narratives | 5,386 |
| Narrative share | 2.1% |
| Companies | 1,383 |
| Products | 12 |
| Issues | 87 |
| States / territories | 60 |
| Compressed Parquet size | 6.09 MB |
| Pipeline runtime | 48.37 seconds |
| NLP topics built | 12 |
| Topic-assigned narratives | 5,343 |
| AI opportunities identified | 30 |
| RAG evaluation questions | 12 |
| Evidence-search citation check pass rate | 100.0% |
| Routing model baseline macro F1 | 0.052 |
| Routing model macro F1 | 0.729 |

## Source Metadata

```json
{
  "source": "CFPB Consumer Complaint Database",
  "source_url": "https://files.consumerfinance.gov/ccdb/complaints.csv.zip",
  "retrieved_at_utc": "2026-08-09T22:16:52.051763+00:00",
  "raw_zip_path": "/private/tmp/Project_resume_launch/data/raw/complaints.csv.zip",
  "output_path": "/private/tmp/Project_resume_launch/data/processed/cfpb_complaints.parquet",
  "record_count": 250000,
  "date_min": "2016-03-02",
  "date_max": "2026-08-09",
  "narrative_count": 5386,
  "runtime_seconds": 48.37,
  "limited_extract": true
}
```

## Interpretation

This run uses a capped official CFPB extract (`limited_extract=true`) for practical local execution. The engineering path supports full bulk refresh through the same command with the cap removed.
