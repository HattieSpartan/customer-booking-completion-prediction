# Predicting customer bookings (job simulation)

A random forest that predicts whether a customer completes a booking, and a check on which booking details carry the signal. Built as Task 2 of a British Airways data science job simulation on Forage. This is a learning exercise, not client work, and it was not reviewed by British Airways.

## Question

From the details of a booking enquiry (country, trip length, lead time, extras requested and so on), how well can we predict whether the customer completes the booking, and which variables matter?

## Result

| Measure | Value |
|---|---|
| Enquiries | 50,000 |
| Completed bookings | 15.0% |
| Accuracy of always predicting "no booking" | 85.0% |
| ROC AUC, 5-fold cross-validation | 0.766 (fold spread 0.004) |
| PR AUC (chance level is 0.150) | 0.357 |
| At a 0.5 cut-off: flagged / recall / precision | 24.5% / 54.9% / 33.5% |

- Accuracy is the wrong measure: a model that never predicts a booking is 85% accurate.
- Booking country dominates. Shuffling it lowers held-out ROC AUC by 0.190; no other variable lowers it by more than 0.023. Dropping booking country from the model lowers ROC AUC from 0.766 to 0.679.
- Completion rate is 5.0% for bookings from Australia (17,872 rows) and 34.4% from Malaysia (7,174). See `results/origin_rates.csv`.
- Day of week, flight hour, meals and weekend flights add almost nothing.

![Feature importance](results/feature_importance.png)

## Method

- Stratified 5-fold cross-validation, so each fold keeps the 15% booking rate.
- Random forest (class weights balanced), 200 trees, minimum 5 rows per leaf.
- Importance is permutation importance on held-out folds, measured as the drop in ROC AUC. All dummy columns for one variable (such as country) are shuffled together, so each original variable gets one number. The forest's built-in importance is saved too (`impurity_mean` in `results/feature_importance.csv`); it ranks lead time and length of stay far higher than permutation does, which is a known bias towards variables with many distinct values.
- Features added: weekend flight, number of extras requested. Booking country is limited to the 15 most common countries plus "Other". Route is dropped (799 routes, most with few rows); flight duration carries route length.

## Limits

- Booking country is a strong predictor, but this data cannot say why. It could reflect markets, pricing, partners or how the data was collected, so it should not be read as cause.
- 719 rows are exact duplicates and 16 of those conflict on the label. Removing duplicates changes ROC AUC from 0.766 to 0.762, so they are kept.
- Cross-validation here is random, not by time. A model used for real decisions should be tested on a later period.
- At the default cut-off, only about one in three flagged enquiries completes. Use it to prioritise follow-up, not to decide individual cases.

## Run it

```
pip install -r requirements.txt
python booking_model.py path/to/customer_booking.csv --out-dir results
```

The dataset belongs to the Forage simulation and is not included in this repository.

## Files

- `BA_Task_2_Customer_Booking_Prediction.ipynb`: the analysis as a notebook with commentary (run it in Colab)
- `booking_model.py`: data preparation, cross-validation, importance, charts
- `results/`: summary metrics, fold metrics, threshold table, importance table and chart, completion rate by country
- `requirements.txt`

## Author

Harriet Joseph. [LinkedIn](https://www.linkedin.com/in/harriet-joseph-5669b921b)
