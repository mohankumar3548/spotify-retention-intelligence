# Spotify Churn Radar: results

Built for Spotify. Prepared by K Mohan Kumar.

## Key figures

- **90-day churn:** 14.0% (420 of 3,000 subscribers)
- **Active users at risk:** 334,000 (Score >= 35%, scaled to 6M base)
- **Revenue at risk:** $52.7M (Annual subscription value)
- **Revenue saved:** $13.2M (At 25% play success)
- **Model AUC:** 0.86 (1.0 = perfect, 0.5 = coin flip)

## Algorithm, as it ran

1. **Collect subscriber data**: 3,000 subscribers, 14.0% churned in 90 days
2. **Engineer behaviour features**: 10 behaviour features per subscriber
3. **Train churn classifier**: Test AUC 0.858 on 900 held-out users
4. **Explain each user**: Top driver: Listening change (90d)
5. **Time to churn**: Median time to cancel 2.2 months
6. **Segment at-risk users**: 4 segments, 167 of the sample at risk
7. **Text signals**: Top complaint theme among churners: Price
8. **Retention plays**: 4 plays assigned

## Recommended actions

1. **Disengaged listeners: Re-engagement.** 224,000 users at 50% average risk, about 2.1 months from cancelling. a fresh Daily Mix and a push for their most played artist. *Impact: $9.0M/yr saved*
2. **Payment friction: Payment rescue.** 80,000 users at 50% average risk, about 2.1 months from cancelling. retry on payday, card update reminder, 7-day grace period. *Impact: $3.0M/yr saved*
3. **Price-sensitive: Plan-switch offer.** 24,000 users at 40% average risk, about 2.6 months from cancelling. Duo or annual plan at a lower monthly cost. *Impact: $923K/yr saved*
4. **Service issues: Support recovery.** 6,000 users at 43% average risk, about 2.4 months from cancelling. priority callback and a free month for open tickets. *Impact: $216K/yr saved*
5. **Stop discounting users the model calls safe.** 4,128,000 users score under 15%. Take them out of win-back offers and keep the budget for the segments above. *Impact: Lower CAC*

_Generated from sample data. Figures are illustrative._
