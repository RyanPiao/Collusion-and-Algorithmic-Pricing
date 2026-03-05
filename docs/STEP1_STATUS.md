# Step 1 Status

## Scope completed
- Established core project framing around algorithmic pricing and competition effects in Airbnb.
- Consolidated cleaning and preprocessing workflow across calendar/listings inputs.
- Set up the main empirical notebook (`fuzzy_rdd_boston.ipynb`) with policy-date cutoff logic and fuzzy-treatment workflow.

## Data and preprocessing progress
- Source ingestion and city-level cleaning routines are implemented in `Step_1_Data_Cleaning.ipynb`.
- Calendar-specific processing and merged exports are implemented in `calendar_data_cleaning.ipynb`.
- Additional grouped/aggregated transformations are staged in `Step_2_Data_Clustering.ipynb`.

## Identification setup (implemented)
- Cutoff markers included for Smart Pricing timing diagnostics.
- Running-variable construction and bandwidth filtering present.
- Eligibility proxy operationalized via `available == 1`.
- Fuzzy adoption intensity represented through propensity-style modeling from host/listing covariates.

## Diagnostics already in notebook
- Bandwidth sensitivity loop.
- Multicollinearity checks.
- Heteroskedasticity test.
- Functional-form and residual diagnostics.

## Pending before full empirical write-up
- Freeze final sample definition and event window.
- Standardize final model specs for table-ready reporting.
- Add clearer export pipeline for reproducible figures/tables.
