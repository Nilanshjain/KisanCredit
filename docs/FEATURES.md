# KisanCredit - Feature Engineering Guide

> Comprehensive explanation of all 45 features used for loan prediction

---

## Overview

The ML model uses **45 features** across **6 categories** to predict loan profitability. Each feature captures a specific aspect of financial behavior, social connections, or location patterns.

**Categories:**
1. Income Features (9) - Earnings and income patterns
2. Expense Features (9) - Spending habits and savings
3. Social Features (8) - Contact network and relationships
4. Discipline Features (6) - Financial responsibility
5. Behavioral Features (6) - Risk indicators and digital engagement
6. Location Features (7) - Geographic stability and mobility

---

## Feature Categories

### 1. Income Features (9 Features)

**Purpose:** Understand earning capacity, consistency, and growth potential.

#### `income_monthly_avg` (float, > 0)
**What it is:** Average monthly income in ₹ over last 6 months

**How it's calculated:**
```python
income_monthly_avg = total_income_6_months / 6
```

**Example Values:**
- ₹15,000: Low income (daily wage worker)
- ₹35,000: Middle income (farmer with regular harvest)
- ₹75,000: High income (successful small business)

**Why it matters:** Higher income = higher loan repayment capacity

**Code location:** `src/features/income_features.py`

---

#### `income_consistency_score` (float, 0-1)
**What it is:** How consistent income is month-to-month (1 = very consistent, 0 = highly variable)

**How it's calculated:**
```python
# Standard deviation of monthly income
std_dev = np.std(monthly_incomes)
mean_income = np.mean(monthly_incomes)

# Coefficient of variation
cv = std_dev / mean_income

# Convert to consistency score (inverse of variability)
income_consistency_score = 1 / (1 + cv)
```

**Example Values:**
- 0.9: Salaried employee (same amount every month)
- 0.7: Farmer (seasonal variation but predictable)
- 0.3: Day laborer (highly variable income)

**Why it matters:** Consistent income = reliable loan repayment

---

#### `income_growth_trend` (float, -1 to 1)
**What it is:** Income growth rate over time (-1 = declining, 0 = stable, +1 = growing fast)

**How it's calculated:**
```python
# Linear regression slope
from scipy.stats import linregress
months = [1, 2, 3, 4, 5, 6]
slope, intercept, r_value, p_value, std_err = linregress(months, monthly_incomes)

# Normalize to -1 to 1 range
income_growth_trend = np.clip(slope / mean_income, -1, 1)
```

**Example Values:**
- +0.15: Growing business (15% monthly growth)
- 0: Stable salaried job
- -0.10: Declining income (economic hardship)

**Why it matters:** Growing income = future repayment capacity improving

---

#### `income_source_diversity` (int, 1-10)
**What it is:** Number of different income sources

**How it's calculated:**
```python
# Count unique income categories in SMS/UPI
income_sources = {
  'salary', 'business_revenue', 'rental_income',
  'agriculture', 'freelance', 'government_benefits'
}

income_source_diversity = len(detected_sources)
```

**Example Values:**
- 1: Single job (salary only)
- 3: Farmer + side business + rental
- 5: Multiple businesses and investments

**Why it matters:** Diverse income = lower risk (if one source fails, others continue)

---

#### `income_credit_ratio` (float, 0-1)
**What it is:** Proportion of income received via formal channels (UPI, bank transfer)

**How it's calculated:**
```python
formal_income = sum(upi_credits + bank_transfers)
total_income = formal_income + cash_estimates

income_credit_ratio = formal_income / total_income
```

**Example Values:**
- 0.95: Almost all income via UPI/bank (digitally savvy)
- 0.60: Mix of digital and cash
- 0.30: Mostly cash-based economy

**Why it matters:** Digital income = verifiable, traceable, trustworthy

---

#### `income_seasonal_variance` (float, 0-1)
**What it is:** How much income varies by season (0 = no seasonality, 1 = extreme seasonality)

**How it's calculated:**
```python
# Compare income by season
summer_income = mean(income_apr_may_jun)
monsoon_income = mean(income_jul_aug_sep)
winter_income = mean(income_oct_nov_dec)

# Variance across seasons
seasonal_variance = np.std([summer, monsoon, winter]) / mean_income
income_seasonal_variance = np.clip(seasonal_variance, 0, 1)
```

**Example Values:**
- 0.05: Non-seasonal (salaried job)
- 0.30: Moderate seasonality (retail business)
- 0.80: High seasonality (agricultural harvest)

**Why it matters:** High seasonality = need flexible repayment schedule

---

#### `income_regularity_score` (float, 0-1)
**What it is:** How regularly income arrives (1 = every month on time, 0 = unpredictable)

**How it's calculated:**
```python
# Check if income arrives expected dates
expected_dates = [5, 10, 15, 20, 25]  # Common salary dates
actual_dates = extract_credit_dates()

regularity_score = calculate_date_consistency(expected_dates, actual_dates)
```

**Example Values:**
- 0.95: Salary on 1st of every month
- 0.70: Business income (varies by 5-7 days)
- 0.40: Irregular freelance payments

**Why it matters:** Regular income = predictable cash flow for EMI

---

#### `income_upi_percentage` (float, 0-1)
**What it is:** Percentage of income received via UPI (PhonePe, Google Pay, Paytm)

**How it's calculated:**
```python
upi_income = sum(transactions marked as UPI)
total_income = all_income_sources

income_upi_percentage = upi_income / total_income
```

**Example Values:**
- 0.90: Modern digital user (most income via UPI)
- 0.50: Mix of UPI and traditional banking
- 0.20: Primarily cash/cheque

**Why it matters:** UPI income is instantly verifiable and traceable

---

#### `income_largest_transaction` (float, > 0)
**What it is:** Largest single income transaction in last 6 months (₹)

**How it's calculated:**
```python
income_largest_transaction = max(all_credit_transactions)
```

**Example Values:**
- ₹8,000: Monthly salary
- ₹150,000: Crop sale proceeds
- ₹500,000: Property sale or large contract

**Why it matters:** Indicates earning capacity and business scale

---

### 2. Expense Features (9 Features)

**Purpose:** Understand spending behavior, financial discipline, and savings capacity.

#### `expense_monthly_avg` (float, > 0)
**What it is:** Average monthly expenses in ₹

**Calculation:**
```python
expense_monthly_avg = total_expenses_6_months / 6
```

**Example:** ₹15,000, ₹30,000, ₹60,000

**Why it matters:** Lower expenses relative to income = more savings for EMI

---

#### `expense_to_income_ratio` (float, 0-1)
**What it is:** Proportion of income spent on expenses

**Calculation:**
```python
expense_to_income_ratio = expense_monthly_avg / income_monthly_avg
```

**Example Values:**
- 0.40: Saves 60% of income (excellent)
- 0.70: Saves 30% (good)
- 0.95: Barely saving (risky)

**Why it matters:** Lower ratio = more room for loan repayment

---

#### `expense_essential_ratio` (float, 0-1)
**What it is:** Proportion of expenses on essentials (food, rent, utilities)

**Calculation:**
```python
essential_expenses = food + rent + utilities + medicines
essential_ratio = essential_expenses / total_expenses
```

**Example Values:**
- 0.80: Most spending on essentials (responsible)
- 0.50: Balanced essential/discretionary
- 0.30: High discretionary spending

**Why it matters:** High essential ratio = less flexibility to cut spending if needed

---

#### `expense_luxury_ratio` (float, 0-1)
**What it is:** Proportion spent on luxury/entertainment

**Calculation:**
```python
luxury_expenses = entertainment + dining_out + travel + shopping
luxury_ratio = luxury_expenses / total_expenses
```

**Example Values:**
- 0.05: Minimal luxury spending (frugal)
- 0.15: Moderate lifestyle
- 0.40: High lifestyle expenses

**Why it matters:** High luxury spending = can cut back if loan repayment needed

---

#### `expense_savings_potential` (float, 0-1)
**What it is:** Proportion of income available for savings

**Calculation:**
```python
expense_savings_potential = (income - expenses - existing_loans) / income
```

**Example Values:**
- 0.40: Can save 40% (excellent)
- 0.20: Can save 20% (good)
- 0.05: Minimal savings (tight budget)

**Why it matters:** Higher potential = easier to afford new EMI

---

#### `expense_debt_burden` (float, 0-1)
**What it is:** Existing debt payments as proportion of income

**Calculation:**
```python
existing_emis = sum(all_loan_payments_monthly)
expense_debt_burden = existing_emis / income_monthly_avg
```

**Example Values:**
- 0.10: Low debt (10% to EMIs)
- 0.30: Moderate debt
- 0.60: High debt (risky)

**Why it matters:** Lower burden = more capacity for new loan

---

#### `expense_volatility` (float, 0-1)
**What it is:** How much expenses vary month-to-month

**Calculation:**
```python
cv = np.std(monthly_expenses) / np.mean(monthly_expenses)
expense_volatility = np.clip(cv, 0, 1)
```

**Example Values:**
- 0.10: Consistent spending (disciplined)
- 0.30: Moderate variation
- 0.70: Highly variable (impulsive spending)

**Why it matters:** Low volatility = predictable budget planning

---

#### `expense_category_diversity` (int, 1-20)
**What it is:** Number of different spending categories

**Calculation:**
```python
categories = {food, rent, utilities, transport, entertainment, ...}
expense_category_diversity = len(detected_categories)
```

**Example Values:**
- 5: Basic needs only
- 10: Typical household
- 18: Diverse spending (many interests/needs)

**Why it matters:** More categories = more potential to cut spending if needed

---

#### `expense_bill_timeliness` (float, 0-1)
**What it is:** Proportion of bills paid on time

**Calculation:**
```python
on_time_bills = count(paid_before_due_date)
total_bills = count(all_bills)
expense_bill_timeliness = on_time_bills / total_bills
```

**Example Values:**
- 0.98: Almost always on time
- 0.75: Occasionally late
- 0.50: Frequently late (red flag)

**Why it matters:** Bill payment discipline predicts loan discipline

---

### 3. Social Features (8 Features)

**Purpose:** Assess social stability, support network, and community ties.

#### `social_network_strength` (float, 0-1)
**What it is:** Overall quality of contact network

**Calculation:**
```python
# Weighted score based on contact types
family_weight = 0.4
business_weight = 0.3
govt_weight = 0.3

social_network_strength = (
  (family_size / 50) * family_weight +
  (business_contacts / 100) * business_weight +
  (govt_contacts / 10) * govt_weight
)
```

**Example Values:**
- 0.85: Strong network (family + business + govt)
- 0.60: Moderate network
- 0.30: Weak network

**Why it matters:** Strong network = social pressure to repay + potential support

---

#### `social_total_contacts` (int, >= 0)
**What it is:** Total phone contacts

**Calculation:**
```python
social_total_contacts = len(phone_contacts)
```

**Example Values:**
- 150: Minimal contacts
- 350: Average
- 800: Very social

**Why it matters:** More contacts = more integrated in community

---

#### `social_family_size` (int, >= 0)
**What it is:** Number of family member contacts

**Calculation:**
```python
# Contacts marked as family or with family indicators
family_keywords = ['mom', 'dad', 'brother', 'sister', 'wife', 'husband']
social_family_size = count(matching_contacts)
```

**Example Values:**
- 15: Small family
- 30: Extended family
- 60: Large joint family

**Why it matters:** Family support network for emergencies

---

#### `social_business_contacts` (int, >= 0)
**What it is:** Number of business-related contacts

**Calculation:**
```python
# Contacts with business indicators
business_keywords = ['shop', 'supplier', 'customer', 'vendor']
social_business_contacts = count(matching_contacts)
```

**Example Values:**
- 20: Small business
- 75: Established business
- 200: Large business network

**Why it matters:** Business network = income stability

---

#### `social_government_contacts` (int, >= 0)
**What it is:** Number of government/bank/official contacts

**Calculation:**
```python
govt_keywords = ['bank', 'post office', 'government', 'office', 'official']
social_government_contacts = count(matching_contacts)
```

**Example Values:**
- 2: Minimal formal connections
- 5: Typical (bank, post office)
- 12: Well-connected (multiple govt offices)

**Why it matters:** Govt contacts = formal sector integration

---

#### `social_communication_frequency` (int, >= 0)
**What it is:** Average calls/SMS per month

**Calculation:**
```python
total_communications = calls + sms_sent + sms_received
social_communication_frequency = total_communications / 6  # per month
```

**Example Values:**
- 30: Low communication
- 60: Average
- 150: Very social

**Why it matters:** Active communication = socially engaged

---

#### `social_contact_diversity` (float, 0-1)
**What it is:** Diversity of contact types

**Calculation:**
```python
# Entropy of contact types
contact_types = {family, business, govt, friends, other}
proportions = [count(type) / total for type in contact_types]
entropy = -sum(p * log(p) for p in proportions if p > 0)
social_contact_diversity = entropy / log(len(contact_types))
```

**Example Values:**
- 0.9: Balanced diverse network
- 0.6: Moderate diversity
- 0.3: Concentrated (mostly one type)

**Why it matters:** Diverse network = resilient support system

---

#### `social_network_depth` (int, 1-10)
**What it is:** Levels of connections (friends of friends)

**Calculation:**
```python
# Analyze mutual contacts
level_1 = direct_contacts
level_2 = contacts_of_contacts
level_3 = contacts_of_level_2
social_network_depth = max_reachable_level
```

**Example Values:**
- 2: Basic network
- 4: Well-connected
- 7: Hub in community

**Why it matters:** Deep network = strong community ties

---

### 4. Discipline Features (6 Features)

**Purpose:** Measure financial responsibility and payment discipline.

#### `discipline_overall_score` (float, 0-1)
**What it is:** Composite financial discipline score

**Calculation:**
```python
discipline_overall_score = (
  discipline_emi_regularity * 0.3 +
  discipline_bill_payment_score * 0.3 +
  (1 - discipline_failed_transactions/10) * 0.2 +
  (1 - discipline_overdraft_frequency/5) * 0.1 +
  discipline_savings_consistency * 0.1
)
```

**Example Values:**
- 0.90: Excellent discipline
- 0.70: Good discipline
- 0.40: Poor discipline

**Why it matters:** Past discipline predicts future loan behavior

---

#### `discipline_emi_regularity` (float, 0-1)
**What it is:** Historical loan payment regularity

**Calculation:**
```python
on_time_emis = count(paid_by_due_date)
total_emis = count(all_past_emis)
discipline_emi_regularity = on_time_emis / total_emis
```

**Example Values:**
- 0.98: Always on time
- 0.80: Occasionally late (1-2 days)
- 0.60: Frequently late

**Why it matters:** Best predictor of future loan performance

---

#### `discipline_bill_payment_score` (float, 0-1)
**What it is:** Utility bill payment discipline

**Calculation:**
```python
on_time_bills = count(electricity + water + phone paid before due)
total_bills = count(all_bills)
discipline_bill_payment_score = on_time_bills / total_bills
```

**Example Values:**
- 0.95: Excellent
- 0.75: Good
- 0.50: Poor

**Why it matters:** Bill discipline correlates with loan discipline

---

#### `discipline_failed_transactions` (int, >= 0)
**What it is:** Number of failed transactions (insufficient funds) in 6 months

**Calculation:**
```python
discipline_failed_transactions = count(transactions_marked_failed)
```

**Example Values:**
- 0: No failures (excellent)
- 2: Rare failures
- 8: Frequent failures (red flag)

**Why it matters:** Failed transactions = cash flow problems

---

#### `discipline_overdraft_frequency` (int, >= 0)
**What it is:** Times account went negative

**Calculation:**
```python
discipline_overdraft_frequency = count(balance < 0)
```

**Example Values:**
- 0: Never overdrawn
- 2: Occasional
- 6: Frequent (high risk)

**Why it matters:** Overdrafts = poor cash management

---

#### `discipline_savings_consistency` (float, 0-1)
**What it is:** Regularity of savings deposits

**Calculation:**
```python
savings_months = count(months_with_savings_increase)
total_months = 6
discipline_savings_consistency = savings_months / total_months
```

**Example Values:**
- 0.90: Saves almost every month
- 0.50: Saves occasionally
- 0.20: Rarely saves

**Why it matters:** Consistent saving = financial discipline

---

### 5. Behavioral Features (6 Features)

**Purpose:** Identify risk factors and assess digital engagement.

#### `behavioral_risk_score` (float, 0-1)
**What it is:** Composite risk indicator (higher = riskier)

**Calculation:**
```python
behavioral_risk_score = (
  behavioral_gambling_indicator * 0.4 +
  (behavioral_location_changes / 10) * 0.2 +
  behavioral_night_transaction_ratio * 0.3 +
  (1 - behavioral_financial_literacy) * 0.1
)
```

**Example Values:**
- 0.05: Very low risk
- 0.25: Moderate risk
- 0.70: High risk

**Why it matters:** Risk indicators predict default probability

---

#### `behavioral_gambling_indicator` (int, 0 or 1)
**What it is:** Presence of gambling transactions (binary)

**Calculation:**
```python
gambling_keywords = ['bet', 'lottery', 'casino', 'game of chance']
behavioral_gambling_indicator = 1 if any_detected else 0
```

**Values:**
- 0: No gambling detected
- 1: Gambling detected

**Why it matters:** Gambling = financial irresponsibility risk

---

#### `behavioral_location_changes` (int, >= 0)
**What it is:** Number of residence changes in last 2 years

**Calculation:**
```python
# Detect location changes from GPS/address updates
behavioral_location_changes = count(distinct_primary_locations)
```

**Example Values:**
- 0-1: Stable residence
- 3: Some mobility
- 7+: Very transient (risk factor)

**Why it matters:** Frequent moves = instability

---

#### `behavioral_night_transaction_ratio` (float, 0-1)
**What it is:** Proportion of transactions at night (10pm-6am)

**Calculation:**
```python
night_transactions = count(transactions between 22:00-06:00)
total_transactions = count(all_transactions)
behavioral_night_transaction_ratio = night_transactions / total
```

**Example Values:**
- 0.05: Normal daytime activity
- 0.15: Some night activity
- 0.40: Mostly night activity (unusual pattern)

**Why it matters:** High night activity can indicate irregular lifestyle

---

#### `behavioral_financial_literacy` (float, 0-1)
**What it is:** Understanding of financial concepts

**Calculation:**
```python
# Inferred from:
# - Use of financial products (mutual funds, insurance)
# - Diversified savings
# - Tax-saving instruments
# - Investment apps used

financial_products_count = count(detected_products)
behavioral_financial_literacy = min(financial_products_count / 10, 1.0)
```

**Example Values:**
- 0.85: High literacy (uses multiple products)
- 0.60: Moderate literacy
- 0.30: Low literacy (basic banking only)

**Why it matters:** Financial literacy = better money management

---

#### `behavioral_app_usage_score` (float, 0-10)
**What it is:** Digital engagement level

**Calculation:**
```python
# Count usage of:
# - UPI apps (PhonePe, Google Pay, Paytm)
# - Banking apps
# - Investment apps
# - Bill payment apps

behavioral_app_usage_score = count(active_financial_apps)
```

**Example Values:**
- 2: Minimal (basic banking only)
- 6: Moderate (UPI + 1-2 apps)
- 9: High (multiple financial apps)

**Why it matters:** Digital engagement = modern financial behavior

---

### 6. Location Features (7 Features)

**Purpose:** Assess geographic stability and urban/rural context.

#### `location_stability_score` (float, 0-1)
**What it is:** How stable residence location is

**Calculation:**
```python
# Variance in GPS coordinates over time
location_variance = np.var(gps_coordinates)
location_stability_score = 1 / (1 + location_variance)
```

**Example Values:**
- 0.95: Very stable (same area always)
- 0.70: Moderate (some travel)
- 0.40: Unstable (frequent moves)

**Why it matters:** Stable location = easier to contact/recover loan

---

#### `location_mobility_score` (float, 0-1)
**What it is:** How much person travels

**Calculation:**
```python
travel_distance = sum(distances_between_locations)
avg_distance_per_month = travel_distance / 6
location_mobility_score = min(avg_distance_per_month / 1000, 1.0)
```

**Example Values:**
- 0.10: Low mobility (stays local)
- 0.40: Moderate travel
- 0.80: High mobility (travels frequently)

**Why it matters:** High mobility = harder to locate if default

---

#### `location_travel_frequency` (int, >= 0)
**What it is:** Number of trips per month

**Calculation:**
```python
trips = count(location_changes > 50km)
location_travel_frequency = trips / 6
```

**Example Values:**
- 2: Occasional travel
- 5: Regular travel
- 15: Frequent traveler

**Why it matters:** Context for mobility score

---

#### `location_distance_from_center` (float, >= 0)
**What it is:** Distance from nearest urban center (km)

**Calculation:**
```python
nearest_city = find_closest_urban_center(gps_location)
location_distance_from_center = calculate_distance(location, nearest_city)
```

**Example Values:**
- 5 km: Urban/suburban
- 25 km: Semi-rural
- 80 km: Remote rural

**Why it matters:** Rural areas have different income/expense patterns

---

#### `location_urban_score` (float, 0-1)
**What it is:** How urban the location is (0 = rural, 1 = metro)

**Calculation:**
```python
# Based on:
# - Population density
# - Infrastructure (hospitals, banks, schools nearby)
# - Digital connectivity

location_urban_score = (
  population_density_score * 0.4 +
  infrastructure_score * 0.3 +
  connectivity_score * 0.3
)
```

**Example Values:**
- 0.9: Metro city (Mumbai, Delhi)
- 0.6: Town (district headquarters)
- 0.2: Village (rural)

**Why it matters:** Urban vs rural affects income/expense patterns

---

#### `location_unique_places` (int, >= 0)
**What it is:** Number of unique locations visited

**Calculation:**
```python
location_unique_places = count(distinct_gps_clusters)
```

**Example Values:**
- 5: Home + work + few places
- 10: Moderate variety
- 25: Many different places

**Why it matters:** More places = more social/business connections

---

#### `location_consistency_score` (float, 0-1)
**What it is:** Regularity of location patterns

**Calculation:**
```python
# Check if same places visited regularly
regular_places = count(places_visited_weekly)
total_places = location_unique_places
location_consistency_score = regular_places / total_places
```

**Example Values:**
- 0.90: Very regular (home-work-home)
- 0.60: Moderate regularity
- 0.30: Unpredictable movement

**Why it matters:** Regular patterns = stable lifestyle

---

## Feature Importance (Model's Perspective)

Based on SHAP values, here are the top 15 most important features:

| Rank | Feature | Importance | Why |
|------|---------|------------|-----|
| 1 | `income_consistency_score` | 0.18 | Stable income = reliable repayment |
| 2 | `expense_to_income_ratio` | 0.15 | Lower ratio = more savings |
| 3 | `discipline_emi_regularity` | 0.12 | Past loan behavior predicts future |
| 4 | `income_monthly_avg` | 0.10 | Earning capacity matters |
| 5 | `expense_savings_potential` | 0.08 | Room for new EMI |
| 6 | `social_network_strength` | 0.07 | Social pressure to repay |
| 7 | `discipline_overall_score` | 0.06 | Financial responsibility |
| 8 | `behavioral_risk_score` | 0.05 | Risk indicators |
| 9 | `income_growth_trend` | 0.04 | Future earning potential |
| 10 | `location_stability_score` | 0.03 | Easy to contact |
| 11 | `expense_bill_timeliness` | 0.03 | Payment discipline |
| 12 | `discipline_failed_transactions` | 0.02 | Cash flow issues |
| 13 | `income_upi_percentage` | 0.02 | Digital verification |
| 14 | `behavioral_financial_literacy` | 0.02 | Money management |
| 15 | `social_total_contacts` | 0.01 | Community integration |

**Total explained variance:** ~98%

---

## Feature Engineering Best Practices

### 1. Normalization
All scores (0-1 range) are normalized:
```python
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)
```

### 2. Missing Data Handling
```python
# Use median for missing numerical features
if feature is None:
    feature = median_value_from_training_data

# Use mode for categorical
if category is None:
    category = most_common_category
```

### 3. Outlier Treatment
```python
# Cap extreme values
income_monthly_avg = np.clip(income, 0, 500000)  # Cap at ₹5L/month
```

### 4. Feature Interactions
Some features are derived from combinations:
```python
expense_savings_potential = (income - expenses - debt) / income
social_network_strength = weighted_sum(family, business, govt)
```

---

## Code Locations

| Category | File | Lines |
|----------|------|-------|
| Income | `src/features/income_features.py` | ~150 |
| Expense | `src/features/expense_features.py` | ~140 |
| Social | `src/features/social_features.py` | ~130 |
| Discipline | `src/features/discipline_features.py` | ~110 |
| Behavioral | `src/features/behavioral_features.py` | ~120 |
| Location | `src/features/location_features.py` | ~100 |
| Pipeline | `src/features/feature_engineering.py` | ~200 |

---

## Testing Features

To test feature extraction:

```python
# Load test data
from src.pipeline.data_generator import generate_synthetic_application

# Generate test application
application = generate_synthetic_application()

# Extract features
from src.features.feature_engineering import extract_all_features
features = extract_all_features(application)

# Verify feature count
assert len(features) == 45

# Check ranges
for feature_name, value in features.items():
    if '_score' in feature_name or '_ratio' in feature_name:
        assert 0 <= value <= 1, f"{feature_name} out of range: {value}"
```

---

This comprehensive feature set enables the model to make accurate loan decisions without traditional credit scores, serving the 190M credit-invisible Indians.
