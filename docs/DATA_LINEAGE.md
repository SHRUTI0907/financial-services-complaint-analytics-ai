# Data Lineage

## CFPB Complaint Metrics
- Source: CFPB Consumer Complaint Database.
- Official access: `https://files.consumerfinance.gov/ccdb/complaints.csv.zip` and the CFPB Open Data API.
- Original fields: complaint ID, date received, product, sub-product, issue, sub-issue, company, state, ZIP code where available, submission channel, company response, timely response, public narrative, and public company response.
- Transformation: official columns are normalized to snake_case, dates are parsed, complaint IDs are numeric, and large records are cached as Parquet.
- Refresh: user-triggered local pipeline. CFPB notes that the public database generally updates daily.
- Limitation: CFPB states the database is not a statistical sample and should not be treated as representative of all consumer experiences.

## SEC Scale Metrics
- Source: SEC EDGAR XBRL Companyfacts API.
- Official access: `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`.
- Original fields: XBRL concepts such as assets, revenue, net income, and operating expenses where consistently reported.
- Transformation: selected latest USD facts are extracted for manually mapped registrants.
- Refresh: user-triggered SEC fetch. Automated access should include a descriptive User-Agent.
- Limitation: mapped registrant assets/revenue are scale proxies, not customer-count denominators.

## BLS Wage Benchmarks
- Source: BLS Occupational Employment and Wage Statistics.
- Official access: BLS OEWS downloadable national estimates.
- Original fields: SOC code, occupation, hourly wage, annual wage.
- Transformation: selected operational roles are extracted for scenario modeling.
- Refresh: annual release cadence.
- Limitation: wage benchmarks are external labor-cost proxies, not company-specific staffing costs.

## NLP Artifacts
- Source: CFPB public complaint narratives where consumers opted to publish and CFPB redacted personal information.
- Transformation: text cleaning, TF-IDF vectorization, topic discovery, retrieval indexing, and classification.
- Refresh: recomputed after complaint ingestion.
- Limitation: narratives are available only for a subset of complaints.

## Value Model
- Observed input: complaint volume for selected segments.
- External benchmark: BLS hourly wage estimate.
- User assumptions: average handling time, AI-addressable share, time reduction, adoption, implementation cost, operating cost, success probability, and discount rate.
- Model output: estimated annual capacity value, annual net value, 3-year net value, ROI, NPV, and payback.
- Limitation: outputs are scenario estimates, not realized savings.
