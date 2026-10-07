def bayes(prior, likelihood, false_positive_rate):
    evidence = likelihood * prior + false_positive_rate * (1 - prior)
    posterior = likelihood * prior / evidence

    return posterior

result = bayes(prior=0.0001, likelihood=0.99, false_positive_rate=0.01)
print(f"P(sick|positive) = {result:.4f}")

# =====================================================================
# THE COMPLETE COMPREHENSIVE GUIDE TO BAYES' THEOREM IN CODE
# =====================================================================

# ---------------------------------------------------------------------
# PART 1: THE CORE CODE & VARIABLES (WHAT EACH TERM MEANS)
# ---------------------------------------------------------------------

# 'prior' -> P(Sick)
# Baseline probability that a random person has the disease BEFORE taking the test.
# Here, it is 0.0001 (0.01%), meaning only 1 in 10,000 people are actually sick.


# 'likelihood' -> P(Positive | Sick)
# The test's accuracy/sensitivity for sick people.
# The probability that the test returns a positive result GIVEN the person is truly sick.
# Here, it is 0.99 (99%).


# 'false_positive_rate' -> P(Positive | Healthy)
# The test's error rate for healthy people.
# The probability that the test returns a positive result GIVEN the person is actually healthy.
# This is equal to (1 - Specificity). Here, it is 0.01 (1%).


# ---------------------------------------------------------------------
# PART 2: HOW WE GET THE 'EVIDENCE' FORMULA (THE DENOMINATOR)
# ---------------------------------------------------------------------
# The 'evidence' represents P(Positive) -> The total probability of testing positive.
# Derived from the Law of Total Probability, it splits the universe into two paths:
#
# Path A: Truly Sick people who test positive.
#         Formula: likelihood * prior
#         Math: P(Positive | Sick) * P(Sick)
#
# Path B: Healthy people who accidentally test positive (False Alarms).
#         Formula: false_positive_rate * (1 - prior)
#         Math: P(Positive | Healthy) * P(Healthy)
#
# By adding these two mutually exclusive paths together, you get every possible positive test.


# ---------------------------------------------------------------------
# PART 3: HOW WE GET THE 'POSTERIOR' FORMULA (THE FINAL ANSWER)
# ---------------------------------------------------------------------
# The 'posterior' represents P(Sick | Positive).
# The probability that a person is actually sick GIVEN they already tested positive.
#
# Derived from the fundamental definition of conditional probability:
# P(A | B) = P(A and B) / P(B)
#
# In medical terms:
# P(Sick | Positive) = P(Sick and Positive) / P(Positive)
#
# Translating this directly into our variables:
# Numerator: P(Sick and Positive) -> Path A -> (likelihood * prior)
# Denominator: P(Positive)         -> Total Pool -> (evidence)
#
# Visual Concept: Target Slice (True Positives) / Entire Universe (Total Positives)


# ---------------------------------------------------------------------
# PART 4: MATHEMATICAL STEP-BY-STEP CALCULATION
# ---------------------------------------------------------------------
# 1. Calculate Evidence:
#    evidence = (0.99 * 0.0001) + (0.01 * (1 - 0.0001))
#    evidence = 0.000099 + (0.01 * 0.9999)
#    evidence = 0.000099 + 0.009999
#    evidence = 0.010098  <-- Total pool of positive tests
#
# 2. Calculate Posterior:
#    posterior = 0.000099 / 0.010098
#    posterior = 0.0098039...
#
# 3. Final Result:
#    P(sick|positive) = 0.0098 (or about 0.98%)


# ---------------------------------------------------------------------
# PART 5: THE CORE INTUITION (WHY IS THE PROBABILITY SO LOW?)
# ---------------------------------------------------------------------
# Even though the test is 99% accurate, the disease itself is extremely rare.
# Because the population of healthy people is so massive (9,999 out of 10,000),
# a tiny 1% false positive rate still triggers a massive flood of false alarms.
#
# Specifically, out of 10,000 people:
# - Only 1 person is actually sick (and they will test positive).
# - About 100 healthy people will accidentally test positive (1% of 9,999).
#
# Since there are ~101 total positive tests and only 1 belongs to a sick person,
# your individual positive test is far more likely to be a false alarm!



# ============================================================================================
#                     MASTER STUDY NOTES: BAYESIAN PROBABILITY VS. ERROR METRICS
# ============================================================================================
#
# Core Concept:
# In Machine Learning, making a prediction (Probability) and scoring a model's performance
# (Metrics) are completely decoupled operations. They use different formulas, operate at
# different times, and address entirely distinct problems in the software pipeline.
#
# ============================================================================================
# MODULE 1: THE IN-LINE PREDICTOR — BAYESIAN PROBABILITY (THE MICRO VIEW)
# ============================================================================================
#
# 1. WHAT IS IT?
#    - It is the mathematical mechanism the model uses to generate its raw output.
#    - Instead of making an uneducated "blind guess," it processes incoming data points
#      and converts uncertainty into a highly calibrated continuous numerical value (0.0 to 1.0).
#
# 2. WHEN AND WHERE DOES IT OPERATE?
#    - It runs in real-time, synchronously, the exact millisecond a feature is passed in.
#    - It operates at the individual data point level (predicting for "Patient X" or "Email Y").
#
# 3. CORE TERMS AND ARCHITECTURE (Breaking Down the Bayes Math):
#    A. PRIOR [ P(Class) ]:
#       - The historical baseline probability of an event before looking at any fresh data.
#       - Example: If 0.01% of humanity has a virus, the model starts with this structural bias.
#    B. LIKELIHOOD [ P(Data | Class) ]:
#       - The probability that this specific pattern/feature would appear *if* the class is true.
#       - Example: How likely is a 104°F fever *given* the patient is infected?
#    C. EVIDENCE [ P(Data) ]:
#       - Total probability of encountering this data point across the entire population.
#       - Acts as a normalizing denominator to scale the final output properly.
#    D. POSTERIOR [ P(Class | Data) ]:
#       - The end result. Your updated probability score after combining the Prior and Likelihood.


# ============================================================================================
# MODULE 2: THE GLOBAL EVALUATOR — PERFORMANCE METRICS (THE MACRO VIEW)
# ============================================================================================
#
# 1. WHAT IS IT?
#    - It is an architectural supervisor/referee that scores the structural accuracy of a model.
#    - It aggregates thousands of past predictions, compares them against absolute ground truth,
#      and compresses the overall systemic error down into a single human-readable score.
#
# 2. WHEN AND WHERE DOES IT OPERATE?
#    - It runs asynchronously or outside of standard production runtime loops.
#    - It evaluates batches, test splits, validation arrays, or historic logs over days/months.
#
# 3. METRIC TYPES BY PROBLEM DOMAIN:
#    A. REGRESSION METRICS (Continuous values like housing prices, stock valuations, temperatures):
#       - MSE (Mean Squared Error): Squares individual errors to heavily penalize outlier mistakes.
#       - MAE (Mean Absolute Error): Takes absolute difference to find average linear error.
#    B. CLASSIFICATION METRICS (Discrete categories like Spam/Ham, Sick/Healthy):
#       - Accuracy: Percentage of total correct classifications.
#       - Log-Loss (Cross-Entropy): Evaluates the *confidence* of probabilities, ruthlessly
#         penalizing a model if it is 99% confident about a prediction but dead wrong.


# ============================================================================================
# MODULE 3: THE COMPREHENSIVE WORKFLOW COMPARISON (THE CHEAT SHEET)
# ============================================================================================
#
# | Feature              | Bayes Probability / Model Formula      | MSE / Evaluation Metrics       |
# |----------------------|---------------------------------------|--------------------------------|
# | Core Identity        | The Active Player                     | The External Referee           |
# | Scope                | Micro (Processes 1 item at a time)    | Macro (Processes N items at once)|
# | Math Domain          | Probability Theory                    | Statistical Error Analytics    |
# | Operational Goal     | Quantify uncertainty right now        | Measure systemic performance   |
# | Primary Metric Input | Features of a brand-new instance      | Array of historic outputs      |
# | Final Output Type    | A value between 0.0 and 1.0 (Chance)  | A metric score (Error level)   |
#
# ============================================================================================
# MODULE 4: PIPELINE SIMULATION (HOW THEY INTERACT IN A REAL APPLICATION)
# ============================================================================================
#
# Step 1: Initialize baseline clinical assumptions (The Training Phase setup)
# Step 2: Production Run -> Generate raw predictive probabilities using Bayes
# Step 3: Evaluation Phase -> Use a metric to see how bad our errors are
#
# (Note: Since this is classification, Log-Loss or Brier Score is mathematically ideal,
# but we can pass it through an MSE calculator to demonstrate structural mechanics)
