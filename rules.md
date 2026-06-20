AI Trading Competition Rules
Note: All dates and times in this document are in British Summer Time (BST).

1. Competition Positioning
This is an AI / Quant / Hybrid Trading Competition based on simulated funds, real market quotes, and a real liquidity environment. The goal of the competition is not to reward a single extreme bet, but to reward trading systems that can generate returns amid market volatility, manage risk, and possess reproducible logic.

Participants may use quantitative strategies, AI agents, human-assisted judgment, hybrid strategies, self-developed models, or third-party tools. The organizer will not evaluate the subjective intent of any trading strategy, and will rank, eliminate, and review participants solely on the basis of public, objective, and computable metrics.

2. Competition Account Rules
Item	Rule
Account type	Simulated trading account
Initial funds	1,000,000 USD
Maximum leverage	30x (1:30)
Stop-Out Level	30% — positions are force-liquidated when the margin level falls to 30%
Trading environment	Unified market data, order matching, and account conditions
Ranking basis	Account equity, return rate, maximum drawdown, Sharpe ratio, risk discipline
Principal risk	Zero risk to principal
All participants operate within the same market environment. The platform will not individually adjust price feeds based on the trading behavior of any single participant.

3. Competition Asset Scope
The competition covers the following tradable instruments across three categories: 8 forex pairs, 2 precious metals, and 5 cryptocurrencies (15 instruments in total).

Category	Instruments
Forex	AUD/USD, EUR/CHF, EUR/GBP, EUR/USD, GBP/USD, USD/CAD, USD/CHF, USD/JPY
Metals	XAG/USD, XAU/USD
Crypto	BAR/USD, BTC/USD, ETH/USD, SOL/USD, XRP/USD
4. Pre-Competition Preparation and Access Arrangements
15 Jun — Opening. That evening from 17:00 to 20:00, the competition portal and relevant materials will be made accessible to the public. All registered participants may access competition data, including historical data for backtesting, as well as the technical toolkits provided by participating sponsors. From their first login, all participants can view their trading account credentials (usernames and passwords) to familiarize themselves with the interface and operational workflows, though trading remains disabled during this stage.

18 Jun 22:00 — Second Registration Deadline. The official competition is scheduled to commence at 21 Jun 22:00.

All competition-related data, rules, platform permissions, and trading conditions will be fully and equitably accessible to all participants prior to the official launch.

5. Competition Schedule
Date	Phase	Details
15 Jun	Opening / Rules Announcement	17:00-20:00: Access to the competition portal, historical data, and sponsor toolkits opens; from first login, all participants can view their trading credentials to explore the interface (trading disabled).
18 Jun	Registration Deadline	22:00: Second registration deadline.
21 Jun	Official Launch	22:00: Competition begins; all trading accounts initialize with the same initial funds.
22 Jun	Round 1 Conclusion	22:00: Rankings recorded; number of qualifiers TBC; 22:00-23:00: Compliance review & verification.
23 Jun	Round 2 Conclusion	22:00: Rankings recorded; number of qualifiers TBC; 22:00-23:00: Compliance review & verification.
24 Jun	Round 3 Conclusion	22:00: Rankings recorded; number of qualifiers TBC; 22:00-23:00: Compliance review & verification.
24-26 Jun	Final Phase	24 Jun 22:00 - 26 Jun 22:00: Top 100 compete in the Finals.
26 Jun	Post-Finals Audit / Results Audit	22:00-23:00: Anomaly detection, confirmation of final rankings, and review of trading logs and anomalies.
27 Jun	Results Announcement & Awards	Final rankings, official competition highlights, and award ceremony.
6. Data and Platform Access Rules
The organizer provides all participants with historical market data for strategy backtesting, model training, parameter tuning, and execution logic preparation.

The platform is not guaranteed to feature a complete, built-in backtesting engine. Participants may utilize the provided data to conduct independent backtesting and model evaluation within their own environments.

For participants deploying the platform's native AI Agents, basic strategy evaluation and backtesting assistance may be available, though with less flexibility than a bespoke/self-built framework. This option is primarily tailored for participants with limited trading experience.

7. Pricing and Execution Mechanism
Platform quotes aggregate liquidity from multiple brokers and sources, integrated with risk-pricing logic, to establish the final tradable prices.

The organizer will not skew or adjust quotes based on the trading behavior of any individual participant. The bid/ask quotes seen by all participants at the same moment remain consistent.

Trades execute within a simulated environment engineered to replicate real-market liquidity, depth, spreads, and impact cost as closely as possible. Consequently, both market orders and pending orders are subject to market depth, available volume, partial fills, slippage, and market impact.

8. Transparency & Compliance Mechanism
During the elimination phase (21 Jun – 24 Jun), participants have access to near-real-time leaderboards, peer trading logs, current positions, account performance, and risk metrics, subject to a 5-minute latency.

Following the conclusion of each round (between 22:00 and 23:00), the system freezes snapshots for ranking records, compiles trade/risk metrics, and runs anomalous trading detection to review potential compliance violations. Should any anomalies be flagged, the organizer will publicly disclose the anomaly type, the determination criteria, relevant Trade/Order IDs, and the resulting impact on qualification via the official Discord community.

During the final phase, peer trading logs, positions, and the live leaderboard will be blinded. Participants retain full visibility only over their own account equity, active positions, open orders, risk metrics, and available margin.

Upon competition closure on 27 Jun, the organizer will publish the final standings, key performance metrics, verified historical logs, necessary Trade/Order IDs, and official rulings on any penalties or disputes. To ensure regulatory compliance and privacy, all Personally Identifiable Information (PII) will remain strictly protected and undisclosed.

9. Technology Usage & Prize Eligibility
To be eligible for the competition's technology prize, participants are expected to share the technical details of their projects. Following the Round 3 elimination on 24 Jun, eligible participants should provide:

A link to the GitHub repository containing their project code;
An overview of the partner technologies utilized, along with a brief description of their application;
Details regarding their data usage; and
A demonstration showcasing how the project works.
Intellectual Property: Participants retain full ownership and intellectual property rights of their respective projects. Access to a project is requested solely to ensure the fairness and integrity of the judging process, and for no other purpose.

A submission form will be made available on the platform. Further details and instructions will be announced separately.

10. Core Ranking Logic
Final standings are determined by a formula-based composite score: PnL-driven, risk-adjusted, and strictly bound by red-line rules for absolute veto.

No subjective penalties or discretionary deductions shall be applied. Standard rankings are determined strictly via algorithmic formulas; high-risk behaviors are cataloged under Risk Discipline and penalized against explicit quantitative thresholds; critical violations result in direct disqualification or immediate elimination; ambiguous gray-area disputes trigger compliance reviews without resulting in arbitrary point deductions.

Scoring is purely formulaic. Disqualification is binary and rules-driven. Discretionary penalties are strictly zero.

11. Final Score Formula
To eliminate any ambiguity, the composite score is calculated as follows:

F
i
n
a
l
 
S
c
o
r
e
=
70
%
×
R
e
t
u
r
n
 
R
a
n
k
+
15
%
×
D
r
a
w
d
o
w
n
 
R
a
n
k
+
10
%
×
S
h
a
r
p
e
 
R
a
n
k
+
5
%
×
R
i
s
k
 
D
i
s
c
i
p
l
i
n
e
Final Score=70%×Return Rank+15%×Drawdown Rank+10%×Sharpe Rank+5%×Risk Discipline

(Note: "Rank" refers to the percentile or absolute ranking of the specific metric among all active participants.)

12. Metric Calculation Formulas
1. Return
The absolute return for participant 
i
i within the given round is defined as:

R
e
t
u
r
n
i
=
E
q
u
i
t
y
f
i
n
a
l
,
i
−
E
q
u
i
t
y
i
n
i
t
i
a
l
E
q
u
i
t
y
i
n
i
t
i
a
l
Return 
i
​
 = 
Equity 
initial
​
 
Equity 
final,i
​
 −Equity 
initial
​
 
​
 

Variable Definitions:

E
q
u
i
t
y
f
i
n
a
l
,
i
Equity 
final,i
​
 : Total account equity of participant 
i
i at the conclusion of the round.
E
q
u
i
t
y
i
n
i
t
i
a
l
Equity 
initial
​
 : Initial account capital, fixed at 
1
,
000
,
000
 USD
1,000,000 USD.
R
e
t
u
r
n
i
Return 
i
​
 : The net return rate for participant 
i
i.
2. Return Rank
The raw return is converted into a normalized rank score scaled from 0 to 100:

R
e
t
u
r
n
 
R
a
n
k
i
=
100
×
N
−
R
a
n
k
i
N
−
1
Return Rank 
i
​
 =100× 
N−1
N−Rank 
i
​
 
​
 

Ranking Logic: All active, non-eliminated participants are ranked by 
R
e
t
u
r
n
i
Return 
i
​
  in descending order. The resulting rank (
R
a
n
k
i
Rank 
i
​
 ) is then normalized.
Boundary Condition: In the event that only a single active participant remains (
N
=
1
N=1), 
R
e
t
u
r
n
 
R
a
n
k
Return Rank defaults to 100.
3. Maximum Drawdown (MaxDD)
The maximum peak-to-trough decline in account equity during the round is monitored continuously:

M
a
x
D
D
i
=
max
⁡
t
(
P
e
a
k
E
q
u
i
t
y
i
,
t
−
E
q
u
i
t
y
i
,
t
P
e
a
k
E
q
u
i
t
y
i
,
t
)
MaxDD 
i
​
 =max 
t
​
 ( 
PeakEquity 
i,t
​
 
PeakEquity 
i,t
​
 −Equity 
i,t
​
 
​
 )

Variable Definitions:

E
q
u
i
t
y
i
,
t
Equity 
i,t
​
 : Total account equity of participant 
i
i at time 
t
t.
P
e
a
k
E
q
u
i
t
y
i
,
t
PeakEquity 
i,t
​
 : The historical peak equity achieved by participant 
i
i from the inception of the round up to time 
t
t.
M
a
x
D
D
i
MaxDD 
i
​
 : The maximum drawdown recorded for the current round.
4. Drawdown Rank
The maximum drawdown is converted into a normalized rank score scaled from 0 to 100:

D
r
a
w
d
o
w
n
 
R
a
n
k
i
=
100
×
N
−
R
a
n
k
D
D
i
N
−
1
Drawdown Rank 
i
​
 =100× 
N−1
N−RankDD 
i
​
 
​
 

Ranking Logic: All participants are sorted by 
M
a
x
D
D
i
MaxDD 
i
​
  in ascending order (lower drawdown yields a higher score), where 
R
a
n
k
D
D
i
RankDD 
i
​
  represents the participant's absolute position.
5. Sharpe Ratio
This competition utilizes a non-annualized Sharpe Ratio, computed directly from 15-minute account equity returns.

The 15-minute interval return (
r
i
,
t
r 
i,t
​
 ) is calculated as:

r
i
,
t
=
E
q
u
i
t
y
i
,
t
−
E
q
u
i
t
y
i
,
t
−
1
E
q
u
i
t
y
i
,
t
−
1
r 
i,t
​
 = 
Equity 
i,t−1
​
 
Equity 
i,t
​
 −Equity 
i,t−1
​
 
​
 

The Sharpe Ratio for participant 
i
i is defined as:

S
h
a
r
p
e
i
=
Mean
(
r
i
,
t
)
Std
(
r
i
,
t
)
Sharpe 
i
​
 = 
Std(r 
i,t
​
 )
Mean(r 
i,t
​
 )
​
 

Variable Definitions:

r
i
,
t
r 
i,t
​
 : The return achieved by participant 
i
i during the 
t
t-th 15-minute interval.
Mean
(
r
i
,
t
)
Mean(r 
i,t
​
 ): The arithmetic mean of the 15-minute interval returns.
Std
(
r
i
,
t
)
Std(r 
i,t
​
 ): The standard deviation of the 15-minute interval returns.
S
h
a
r
p
e
i
Sharpe 
i
​
 : The non-annualized Sharpe Ratio.
Boundary Constraints:
If 
Std
(
r
i
,
t
)
=
0
Std(r 
i,t
​
 )=0, 
S
h
a
r
p
e
i
Sharpe 
i
​
  is defined as 0.
To prevent statistical anomalies from sparse data, if an account contains fewer than 8 valid 15-minute return observations, its final 
S
h
a
r
p
e
 
R
a
n
k
Sharpe Rank shall be capped at a maximum of 50 points.
6. Sharpe Rank
The Sharpe Ratio is converted into a normalized rank score scaled from 0 to 100:

S
h
a
r
p
e
 
R
a
n
k
i
=
100
×
N
−
R
a
n
k
S
h
a
r
p
e
i
N
−
1
Sharpe Rank 
i
​
 =100× 
N−1
N−RankSharpe 
i
​
 
​
 

Ranking Logic: All active participants are ranked by 
S
h
a
r
p
e
i
Sharpe 
i
​
  in descending order, where 
R
a
n
k
S
h
a
r
p
e
i
RankSharpe 
i
​
  represents the participant's absolute position.
13. Risk Discipline Rules
Each participant starts each round with a baseline Risk Discipline score of 100 points, subject to deductions for verified risk violations, down to a floor of 0. The Risk Discipline score is calculated independently per round and resets automatically at each round's inception.

Critical red-line violations bypass the reset protocol. Actions including forced liquidation, exploitation of system vulnerabilities, API abuse, multi-account participation, or manipulation of competition fairness will, once confirmed, lead directly to disqualification.

1. Margin Usage
The margin utilization rate for participant 
i
i is defined as:

M
a
r
g
i
n
 
U
s
a
g
e
i
=
U
s
e
d
 
M
a
r
g
i
n
i
E
q
u
i
t
y
i
Margin Usage 
i
​
 = 
Equity 
i
​
 
Used Margin 
i
​
 
​
 

Violation Criteria	Risk Discipline Penalty
M
a
r
g
i
n
 
U
s
a
g
e
i
>
90
%
Margin Usage 
i
​
 >90% persisting for a continuous duration of 
≥
30
 minutes
≥30 minutes	-20 points
M
a
r
g
i
n
 
U
s
a
g
e
i
>
95
%
Margin Usage 
i
​
 >95% persisting for a continuous duration of 
≥
15
 minutes
≥15 minutes	-30 points
M
a
r
g
i
n
 
U
s
a
g
e
i
>
98
%
Margin Usage 
i
​
 >98% persisting for a continuous duration of 
≥
10
 minutes
≥10 minutes	Triggers Compliance Review
2. Leverage Usage
The effective leverage ratio for participant 
i
i is calculated as:

L
e
v
e
r
a
g
e
i
=
G
r
o
s
s
 
N
o
t
i
o
n
a
l
 
E
x
p
o
s
u
r
e
i
E
q
u
i
t
y
i
Leverage 
i
​
 = 
Equity 
i
​
 
Gross Notional Exposure 
i
​
 
​
 

Violation Criteria	Risk Discipline Penalty
L
e
v
e
r
a
g
e
i
>
28
x
Leverage 
i
​
 >28x persisting for a continuous duration of 
≥
30
 minutes
≥30 minutes	-20 points
L
e
v
e
r
a
g
e
i
>
29
x
Leverage 
i
​
 >29x persisting for a continuous duration of 
≥
15
 minutes
≥15 minutes	-30 points
L
e
v
e
r
a
g
e
i
Leverage 
i
​
  approaching 
30
x
30x for a continuous duration of 
≥
10
 minutes
≥10 minutes	Triggers Compliance Review
3. Exposure Concentration
Asset and direction concentration metrics are defined via the following allocation ratios:

S
i
n
g
l
e
 
I
n
s
t
r
u
m
e
n
t
 
E
x
p
o
s
u
r
e
i
=
N
o
t
i
o
n
a
l
 
E
x
p
o
s
u
r
e
s
i
n
g
l
e
G
r
o
s
s
 
N
o
t
i
o
n
a
l
 
E
x
p
o
s
u
r
e
i
Single Instrument Exposure 
i
​
 = 
Gross Notional Exposure 
i
​
 
Notional Exposure 
single
​
 
​
 

Violation Criteria	Risk Discipline Penalty
Single-instrument exposure 
>
90
%
>90% persisting for a continuous duration of 
≥
30
 minutes
≥30 minutes	-10 points
Net Directional Exposure 
>
95
%
>95% persisting for a continuous duration of 
≥
30
 minutes
≥30 minutes	-10 points
(Note: Directional trading is permitted. What the rules restrict is the prolonged, extremely concentrated, near-full-leverage use of risk.)

14. Red-Line Rules
Forced Liquidation: Triggers immediate elimination from the competition, with no advancement to the next round.
Exploitation of System Vulnerabilities: Results in immediate disqualification. This includes exploiting system vulnerabilities, erroneous quotes, latency loopholes, matching-engine anomalies, settlement anomalies, or circumventing system limits.
API Abuse: Results in immediate disqualification. This includes maliciously flooding API endpoints, bypassing API rate limits, attacking or interfering with platform services, unauthorized access to systems or data, and high-frequency requests that cause system anomalies.
Safe Harbor Threshold: High-frequency requests within a normal range are not deemed abnormal; for example, requests at or below 500 per second will not be automatically classified as abnormal behavior. However, if request behavior causes system anomalies, circumvents limits, or affects the fairness for other participants, the organizer reserves the right to review.
Multi-Account Participation by the Same User: Results in immediate disqualification. Each participant may use only one account to compete.
Unauthorized Collaboration or Collusion to Manipulate Rankings: Manipulating rankings through multiple accounts, external (out-of-team) collaboration, mutual transfer of risk, pre-arranged trading, or any other means is prohibited.
15. Elimination and Qualification Rules
At the conclusion of each round, the qualification and elimination protocol executes via the following sequential workflow:

22:00 — Data Snapshot: The system freezes and logs all account equity, active positions, historical trading logs, and risk metrics for the current round.
22:00 - 23:00 — Compliance & Audit Window:
Anomaly Detection: Run automated screening for traffic and trading anomalies.
Red-Line Verification: Cross-check trading logs against the defined red-line rules.
Account Purge: Disqualify and remove any accounts that triggered violations or suffered forced liquidation (account wipeout).
Score Calculation & Ranking: Compute individual performance metrics, generate the Final Score for all remaining active accounts, and sort participants in descending order.
Roster Finalization & Public Disclosure: Finalize the official qualification roster for the next round. Any flagged anomalies and compliance rulings will be publicly disclosed within the official Discord community.
Qualification Schedule
Round	Trading Cutoff	Audit & Review Window	Qualified Status
Round 1	22 Jun 22:00	22:00 - 23:00	Qualifiers TBC
Round 2	23 Jun 22:00	22:00 - 23:00	Qualifiers TBC
Round 3	24 Jun 22:00	22:00 - 23:00	Qualifiers TBC
Finals	26 Jun 22:00	22:00 - 23:00	Final Ranking
16. Tie-Breaking Protocols
In the event that multiple participants conclude a round with identical Final Scores, deadlocks will be systematically resolved based on the following strict hierarchy of performance metrics:

Primary: Higher 
R
e
t
u
r
n
i
Return 
i
​
  (descending order).
Secondary: Lower 
M
a
x
D
D
i
MaxDD 
i
​
  (ascending order).
Tertiary: Higher 
S
h
a
r
p
e
i
Sharpe 
i
​
  (descending order).
Quaternary: Higher Risk Discipline score (descending order).
Quinary: More reasonable trading activity.
Fallback: If a tie persists after applying all the above quantitative layers, the organizer will conduct a review and publish the basis for its decision.

17. Best Sharpe Ratio Award
Prize: $10,000 — awarded to the eligible participant with the highest Sharpe Ratio.

Eligibility. To qualify, a participant must:

reach the Finals;
finish within the Top 50 of the final overall ranking;
have no red-line violations;
have executed at least 30 trades.
Sharpe Ratio Formula. The Sharpe Ratio is computed from 15-minute account-equity returns and is not annualized. Returns are sampled at 15-minute intervals throughout the entire competition period.

r
i
,
t
=
E
q
u
i
t
y
i
,
t
−
E
q
u
i
t
y
i
,
t
−
1
E
q
u
i
t
y
i
,
t
−
1
r 
i,t
​
 = 
Equity 
i,t−1
​
 
Equity 
i,t
​
 −Equity 
i,t−1
​
 
​
 

S
h
a
r
p
e
i
=
Mean
(
r
i
,
t
)
Std
(
r
i
,
t
)
Sharpe 
i
​
 = 
Std(r 
i,t
​
 )
Mean(r 
i,t
​
 )
​
 

Variable Definitions:

E
q
u
i
t
y
i
,
t
Equity 
i,t
​
 : Account equity of participant 
i
i at the close of the 
t
t-th 15-minute interval.
E
q
u
i
t
y
i
,
t
−
1
Equity 
i,t−1
​
 : Account equity of participant 
i
i at the close of the previous 15-minute interval.
r
i
,
t
r 
i,t
​
 : The 15-minute account return of participant 
i
i.
Mean
(
r
i
,
t
)
Mean(r 
i,t
​
 ): The arithmetic mean of the 15-minute interval returns.
Std
(
r
i
,
t
)
Std(r 
i,t
​
 ): The standard deviation of the 15-minute interval returns.
S
h
a
r
p
e
i
Sharpe 
i
​
 : The non-annualized Sharpe Ratio of participant 
i
i.
Winner Selection. Among all eligible participants, the participant with the highest Sharpe Ratio shall be declared the winner. In the event of a tie:

The participant with the higher Final Return shall be ranked higher;
If the tie persists, the participant with the lower Maximum Drawdown shall be ranked higher.
18. Appeals and Dispute Resolution
Participants may file reasonable appeals during or after the competition. The organizer will keep feedback and appeal channels open.

For ambiguous gray-area disputes or those difficult to adjudicate objectively, the organizer may disclose the relevant facts, redact personal sensitive information, publish the Trade IDs / Order IDs, organize community discussion, and, where necessary, put the matter to a participant vote.

Scope of Authority Statement: The organizer's role is not to judge the intent behind participants' strategies, but to safeguard the order, fairness, and integrity of the competition.

19. Principle of Public Disclosure of Penalties
For any penalty, elimination, or disqualification decision, the organizer will disclose to all participants:

The reason for the penalty.
The basis for the determination.
The relevant Trade IDs / Order IDs.
The impact on rankings.
For privacy protection, the organizer will not disclose real names, email addresses, phone numbers, identity information, or other personal sensitive information.

20. Organizer's Reserved Rights
The organizer reserves the right to suspend, adjust, review, or modify the competition arrangements in the event of:

System failures.
Market-data anomalies or quote anomalies.
Matching anomalies or settlement anomalies.
API-service anomalies.
Large-scale force majeure or technical issues that clearly affect fairness.
In any such case, the organizer will publish the reasons, scope of impact, and resolution as transparently as possible.

21. Terms & Conditions
These rules provide a summary overview of the competition. Participation is governed by the full, binding Terms & Conditions. Please review the complete document here: Terms & Conditions.


MODEL TO MARKET
The Quantitative Hack

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The UK's first live AI-native trading competition
>
$100,000 Prize Pool  •  300 Participants  •  2 Weeks

Hosted by SYPHONIX × AI ENGINE
📋 EVENT DETAILS

Format: Solo participation (one-person teams only)
Dates: 15 – 27 June 2026
Prize Pool: $100,000 (cash + hardware + AI credits)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 TRADING ENVIRONMENT

• Starting capital: $1,000,000 virtual funds per participant
• Live market simulation: FX, Gold, Silver, Crypto
• Leveraged trading supported
• Paper trading available during the Strategy Prep phase

🎯 TWO WAYS TO TRADE (no coding required)
• MT5 — Trade via MetaTrader 5, the industry-standard platform
• AI Native Interface — Build strategies and trade through a conversational AI interface