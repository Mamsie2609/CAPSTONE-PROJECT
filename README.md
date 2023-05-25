# CAPSTONE-PROJECT

Nexus Bank is a financial institution dedicated to delivering unparalleled banking services to their clients. Their mission is to establish enduring relationships with their customers by providing tailored financial solutions that align with their individual needs and goals.

At Nexus Bank, they believe that every individual deserves access to world-class financial products and services, regardless of their age,  profession, or income level. That's why they offer a wide spectrum of banking solutions to accommodate customers’ lifestyle, including term deposits, personal loans, and mortgage financing.

Their team of seasoned banking professionals are committed to providing customers with the utmost level of service, transparency, and honesty.

Nexus bank has conducted campaigns with the goal of acquiring deposits.
In the last board meeting, the directors where unsatisfied with their current situation and need to optimise the operations at Nexus bank.
The Director of Nexus contacted me because they are interested in leveraging  the power of their data to gain insights into the bank and improve their efficiency. They want to identify patterns and trends in customer behaviour to decipher if customer demographics such as age, educational level etc. influences customers attitude toward defaulting. The board specifically wants to anticipate future customer behaviour and know the likelihood of deposits from customers.

Nexus wants to understand how effective their campaigns are and thus develop marketing campaigns to reach specific customer segments. By analysing customer behaviours, loan trends, and marketing campaign effectiveness, Nexus wants to optimise its operations, mitigate risks / loan defaults, and improve customer deposits. 

# SUMMARY OF FINDINGS

After conducting thorough data wrangling and analysis, the following insights were uncovered:

•	The customer base consists primarily of middle-aged adults, followed by young adults, senior citizens, and the elderly.

•	Blue-collar and management professions have the highest representation among customers, while the dataset includes a smaller number of customers with unknown professions.

•	The customer base is predominantly composed of married individuals, followed by singles, and divorced individuals.

•	Customers with a secondary educational background have the highest representation, followed by tertiary and primary education.

•	The cellular method is the primary mode of communication with customers.

•	Previous marketing campaign outcome: The outcome is mostly unknown, followed by failures and successes.

•	The elderly has the highest balance, followed by senior citizens, while young adults have the lowest balance.

•	Middle-aged adults and young adults have the highest number of customers who have made no deposits.

•	Customers with secondary educational background have the highest number of individuals who have made no deposits.

•	Married individuals have the highest number of customers with no deposits, while divorced individuals have the least.

•	Retired individuals have the highest balance, while the services industry has a smaller representation.

These findings provide valuable insights into the characteristics and behaviours of the customer base, which can help inform decision-making and marketing strategies.

To predict the likelihood of customer deposits, I performed training and testing using various models after normalising the dataset. Considering the specific problem faced by the bank, which is to minimise false positives (incorrectly identifying customers), the focus was on selecting a model with a high precision score. Based on this objective, the SGD classifier, with the highest precision score among the models, was considered the most suitable choice.
