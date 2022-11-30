# Recommender-System-R
 
      In-Course Assessment 
Artificial Intelligence Ethics and Applications CIS4057-N-FJ1-2021
  
NAME: ADEKUNLE ADESEYE
STUDENT NUMBER: B1081572
COURSE: MSC APPLIED DATA SCIENCE




TITLE: IMPLEMENTATION OF MACHINE LEARNING IN AGRICULTURE AND ETHICAL ISSUES ASSOCIATED WITH THE ADOPTION OF MACHINE LEARNING AND DATA SCIENCE TECHNOLOGIES ON FARMLANDS.

Abstract
The future population projections have instigated the interest of big data technology into the agricultural sector as a viable solution that can increase the yield of farming and meet the estimated population growth. Ours research was developed using a mix method of quantitative analysis to analyze the various numeric attributes that can influence the target column and a qualitative analysis of the target column that consist of a categorical list of crops provided by the farmers. A recommendation system was developed using several machine learning algorithms to show the type of crop to be planted in a season based on the weather conditions and soil composition. Ethical concerns were explored, such as supplying the farmers with inaccurate information that could cause a lot of problems, protecting farmers’ information, data ownership and loss of job. We recommended solutions that help build trust with farmers and discussed future works on this research paper. 

Keywords: Machine Learning, Data Science, Farming, Agriculture, Support Vector Machine, K-nearest neighbors, Gradient Boosting Machine, Algorithms. 





Table of Content
Contents
Abstract	2
Introduction	4
Literature Review	5
Research Questions and Hypothesis	5
Methodology	6
Data Selection	6
Data Collection and Cleaning	6
Analysis Approach	6
Quantitative Analysis Overview	8
Analysis and Findings	9
Exploratory Data Analysis	9
Algorithm Selection, Modelling and Prediction	15
Evaluation and Discussion of Findings	16
Ethical Concerns associated with the adoption of Machine learning on Farmlands	17
Conclusion	17
Recommendations	17
Limitations and Future Research	17
Reference	18











Table of Figures
Figure 1. Missing Number Data Profile	6
Figure 2. Label Column Distribution	7
Figure 3. Label Column Distribution	7
Figure 4. Box and Whiskers Plot showing Label Column Vs Attributes	8
Figure 5. Box and Whiskers plot showing label column Vs Attributes	9
Figure 6. Data Info Statistics	9
Figure 7. Percentages	10
Figure 8. Data Structure with Tree Map	10
Figure 9. Descriptive statistics of the numeric variables	11
Figure 10. Descriptive Statistics of the numeric variables	11
Figure 11. Density Plot showing distribution of each attributes	11
Figure 12. Histogram of each Attributes	12
Figure 13. Skewness of each attributes	12
Figure 14. Agostino Results of Humidity	12
Figure 15. Anscombe Test showing the value not equal to 3 for humidity	12
Figure 16. Quantile-Quantile plot by Label	13
Figure 17. qq-plot with histogram to show normality of the attribute	13
Figure 18. qq-plot with histogram to show normality of the attribute	13
Figure 19. Results of Shapiro-Wilk Normality test with p-values	14
Figure 20. Correlation Plot between Attributes	14
Figure 21. Scatterplot with histogram to show statistical details of the attributes	14
Figure 22. Scatterplot with histogram, correlation coefficient and significance star	14
Figure 23. Strength of correlation with scatterplot	15
Figure 24. Visualising the attribute's outliers	15
Figure 25. Table showing the outlier count of each attributes	15
Figure 26. Box and whisker plot with histogram that shows the outlier mean and without outlier mean	16
Figure 27. Line chart to show imputation of the outliers	16
Figure 28. Table showing the removed outlier from K	16
Figure 29. Accuracy statistics for the fitted models	17
Figure 30plot showing the cross-validation and randomly selected predictors	17
Figure 31. Dot plot for each model	17
Figure 32. Overall Selected Random Forest  Model	17






Introduction
Agriculture had been the earliest known endeavor and means to which humans feed themselves. The earliest known evidence of farming practices discovered by archeologist is over 9000 B.C (CITE). Farmers are known to have methods and approaches to achieve their desired outcome, these are belief systems passed on from generations before them (CITE). Farmers are clannish and their methods have shown to be hinderance to mechanized farming through the years. Farming practices have been redundant for a long while, although it has been refined and improved upon through government interventions, with the addition of machinery to facilitate large-scale farming. Even with the mechanized technological strides of most economies, the yields of such practices might not meet the projected goal of sustaining a world population of 9.7 billion by the year 2050 (Agriculture and Food, 2022). It has been proposed that agriculture would be crucial to achieving the world’s development goal of a sustainable, healthy, and inclusive food system (Agriculture and Food, 2022). With the emergence of machine learning, artificial intelligence and data science in the big data space, adoption of these technologies into agriculture seems a viable solution to achieving the vision of a sustainable food security in years to come. A recent approach to the technologicalization of the agricultural sector is the use of machine learning subset known as precision agriculture. This is a farming methodological solution that optimizes the yield of farming by monitoring and observing the changes within a farmland. They use Information collated from multi-spectral imagery analysis of high-resolution satellite images, agricultural drones, and environmental sensors, to create a data framework that responds to changes via analysis of the information provided. The aim of this precision farming framework is to collate and analyze these data, which could be implemented with a machine learning algorithm to create effective intervention or management solutions for the farmland. The focus of this research is to elicit the use and advantages of using some machine learning algorithms in predicting or recommending the type of crop to plant on farmland based on the information acquired from weather, soil, and fertilizers. The research will implement a hybrid methodology, that involves a mix of quantitative and qualitative research methods in collating and analyzing the data from a farmland in India. (Ghadge et al, 2018).




Literature Review
According to (Ghadge et al, 2018), the historically precedence of farming in India had been known to play a crucial role in welfare of the economy and provision of job opportunities to the populace. The bane of most farmers' distress comes from knowing the right crop to plant based on the state and necessity of the soil which may mar the yields of such farmland (Ghadge et al, 2018). Like our intended research, (Gosai et al, 2021) implemented some machine learning algorithms on data provided by IoT sensors for soil nutrients (Nitrogen, Potassium and Phosphorus) and weather conditions (rainfall, humidity, and temperature), to detect the degradation or leeching in soil which could affect farming on a land and crop health. Their results showed that XGBoost achieved 99.3% accuracy.
 Precision Agriculture has thus been used to overcome these challenges by using information gathered from the current composition of the soil, to determine a suitable crop that should be planted for that season (Zhang, Wang and Wang, 2022). The information pertaining to the soil is collated and sent to testing labs, the status from the result is implemented into a recommender system. This process will use ensemble machine learning method with prevalent voting technique. The support vector machine (SVM) and Artificial Neural Network as learners was also used to predict or recommend crops for unique or specific farmlands based on the parameters (soil composition) supplied to the system. The authors used these varieties of machine learning algorithms to predict to the highest degree the accuracy of the recommendations and their efficiency based on the parameters provided. Their paper proposed and showed that using the ensemble method with Majority voting technique, support vector machine and ANN as learners, highest accuracy, and efficiency of the recommendation system was achieved among other machine learning algorithms used to build the model. The algorithms used include Ensemble methods, Support Vector Machine, Naïve Bayes, Random Forest, and Multi-Layer Perceptron. In conclusion the paper showed that the research will help farmers to improve their yield, reduce the disintegration of the soil, nitrogen leaching from nitrogen rich fertilizers and water management. Future works were also proposed to include yield recommendation.
Research Questions and Hypothesis
Furthermore, the research from literature review had led us to ask the question, can we implement machine learning as a recommendation system for farmers and what are the advantages of such systems? Will such systems protect the interest of the farmers? With the risks associated with digital information, will information about the farmers and farmland be secure? We intend to prove that the inclusion of the best fitted machine learning algorithm on such a system will lead to productivity on the farmland, improve yield and reduce nitrogen leeching on the subterranean aquatic life form. We will also propose solutions to the ethical concerns raised with the digitization of farmland. 

Methodology 
Data Selection
A quantitative approach was carried out by augmenting information available on rainfall, temperature, humidity from the government metrological services of India, with other qualitative information from the soil samples sent to laboratories for analysis. A list of crops serving as the qualitative information supplied by the farmers was also added to the overall data. 

Data Collection and Cleaning
The dataset used in analysis and development of this research procedure was from an online source, with the data consisting of 2200 instances and 8 attributes including a labelled column. The attributes show the ratio of the chemical composition of the soil, such as Nitrogen (N), Potassium (K), Phosphorus (P), and Potential of Hydrogen (Ph). Attributes also include weather conditions such as temperature at a given time, amount of rainfall and humidity during each recording. 
 
Figure 1. Missing Number Data Profile
Data cleaning step was done to check for missing and incorrect values. This shows no missing values. A statistical random sampling method without replacement was used to select 1000 instances to facilitate the speed and accuracy of models used in the eventual prediction system.

Analysis Approach
The methodology for data analysis was implemented by using a mix of qualitative and quantitative approaches. Using R language on a RSTUDIO environment, statistical analysis of the quantitative values within the dataset was analyzed, and relationships within the numeric values explored. A qualitative analysis of the labelled column was also implemented.
 
Figure 2. Label Column Distribution
 
Figure 3. Label Column Distribution
The analysis shows the descriptive statistics of every category in the target with a bar chart showing equal distribution in the values of the target column.  
Figure 4. Box and Whiskers Plot showing Label Column Vs Attributes
 
Figure 5. Box and Whiskers plot showing label column Vs Attributes
From the box and whiskers plot, it could be seen that a low to moderate levels of temperature and ph, in the soil composition are required in planting of certain crops while K, N, P and rainfall are required in substantial amounts.
Quantitative Analysis Overview
The quantitative analysis shows an overview of each numeric variable and the type of values they contain. 
 
Figure 6. Data Info Statistics

 
Figure 7. Percentages
This presents the number of rows and columns, the number of categorical and numerical variables, and the percentage of missing values in each of the variables of the data. 
Analysis and Findings
We analyzed the quantitative numeric values (discreet and continuous) within the data. An exploratory data analysis was initiated to explore the values within the data.
Exploratory Data Analysis
 
Figure 8. Data Structure with Tree Map

We start by verifying the variable data structure and size of the dataset objects. This shows the Nitrogen, Phosphorus and Potassium having an integer datatype while temperature, humidity, ph and rainfall have the numeric datatype. Next, we do a quantitative analysis to check the statistics of the attributes in the dataset which shows the counts, mean, median, standard deviation and more. A density and histogram chart showing distribution of each variable was observed to identify the distribution of each variable. Ph and Temperature appears to be gaussian and gaussian like distribution while the rest of the variables are skewed or binomial in nature.  
 
Figure 9. Descriptive statistics of the numeric variables
 
Figure 10. Descriptive Statistics of the numeric variables


 
Figure 11. Density Plot showing distribution of each attributes
 
Figure 12. Histogram of each Attributes
The symmetry of the attributes encouraged a further probe of the skewness and kurtosis of each variable to explore their distribution. By using the skewness function and D’Agostino skewness test (p- value), it appears only ph and temperature are symmetric with the rest being high to moderately skewed.
Using the standard kurtosis value for normal distribution of 3, we used the Anscombe-Glynn kurtosis test that showed values higher than 3 in all the attributes except N, and indication of outliers present in the other variables. 
 
Figure 13. Skewness of each attributes
 
Figure 14. Agostino Results of Humidity
 
Figure 15. Anscombe Test showing the value not equal to 3 for humidity

 The overall distribution of the dataset was viewed in a quantile-quantile plot and a further visualization that includes a histogram of the original data with histogram using log and square root transformation. 
Figure 16. Quantile-Quantile plot by Label
 
Figure 17. qq-plot with histogram to show normality of the attribute
  
Figure 18. qq-plot with histogram to show normality of the attribute

Although qq-plot shows that most of the points are close to the diagonal line indicating normal distribution, the Shapiro-wilk normality test was performed to elicit the size of deviation from normality. 
 
Figure 19. Results of Shapiro-Wilk Normality test with p-values
Next, we explore the relationship between the numeric attributes using correlation plot. There was a slight positive correlation between K and humidity, K and N, and a negative correlation humidity and P. The visualization also shows the P-value of correlation.
 
Figure 20. Correlation Plot between Attributes
 
Figure 21. Scatterplot with histogram to show statistical details of the attributes
 
Figure 22. Scatterplot with histogram, correlation coefficient and significance star
 
Figure 23. Strength of correlation with scatterplot
The scatter and box-whiskers plot were used to compare numeric variables using Pearson correlation and simple linear models.
We checked and removed outliers using the zscore. A statistical summary was also created to show the outlier count and a density plot that visualizes the imputation of the outliers. We also viewed the variable with outlier mean and without mean.
 
Figure 24. Visualising the attribute's outliers
  
Figure 25. Table showing the outlier count of each attributes
 
Figure 26. Box and whisker plot with histogram that shows the outlier mean and without outlier mean
The outlier was imputed with capping method, done by imputing the upper outlier with 95th percentile and the bottom outlier with 5th percentile.
 
Figure 27. Line chart to show imputation of the outliers
 
Figure 28. Table showing the removed outlier from K
Table shows K has been imputed and does not appear on the table and the graph shows the original line chart with the imputed line.
Algorithm Selection, Modelling and Prediction
To continue the research, the data was shuffled to ensure randomness and divided into training and testing set on a ratio of 70% training and 30% testing. A 10-fold cross validation was set to the train control and the training set fitted to 5 models, K nearest neighbor, gradient boosting machine, support vector machine, random forest, and decision tree classifiers. An accuracy metric was chosen, and individual tuning methods were applied. The best fitted model is the random forest with an accuracy metric of 97%.
 
Figure 29. Accuracy statistics for the fitted models
 
Figure 30plot showing the cross-validation and randomly selected predictors
 
Figure 31. Dot plot for each model
  
Figure 32. Overall Selected Random Forest  Model


Evaluation and Discussion of Findings 
For plants to grow, soil mineral uptake needs to regulate the intake of water, ph, macro and micronutrients. Potassium, a mineral classified as macronutrient, is a building block for protein and nucleic acids that helps in crop yield and increase plants quality (Morgan and Connolly, 2013). Thus, in the box and whiskers plot there is a direct relationship between the levels of K, P, N with the classification or type of crop associated with these chemicals (Morgan and Connolly, 2013). Relationships between temperature, rainfall and humidity also affect the crop type. Varying the composition of the data attributes determines the crop type while removal, that was observed when highly correlated feature K was removed resulted in an unwanted performance of the system. The structural visualization using histogram to view the attributes of the dataset gave some non-consistent results with the values gotten from the P-values, Anscombe test and Shapiro Wilk test which implies that the data was truly non-symmetrical data. This allowed us to apply the required algorithm and tests based on that information to produce a highly accurate and efficient system.
Ethical Concerns associated with the adoption of Machine learning on Farmlands 
Smart Information Systems provided by multinational agriculture business, shows that inaccurate information provided by the system to the farmers could have a devastating effect on the earnings, crop production, crop yield and quality of the crop (Ryan, 2019). There is the fear that farmer information could be used against them since sharing such information could fall into the hands of government and regulatory bodies that impose charges on them (Ryan, 2019). They fear that 3rd parties could get their information and sell products back to them for example a fertilizer company could sell them nitrogen enriched products when it knows that their farmland has deficiency in Nitrogen (Ferris, 2017). Automating farming activities and jobs previously reserved for humans for instance agronomists could lead to job losses (Ryan, 2019).
Conclusion
Recommendations
Although the agricultural sector is quite novel in its adoption of big data technology, security concerns relating to hacking and information loss is still nonexistence, hence Agribusiness that provide such technological services should ensure that farmer’s information is protected with regular security updates. Law should be provided in these initial stages to ethical govern the implementation of AI and big data in agriculture, and the ownership and right to sell information should be exclusively reserved for the farmers. 

Limitations and Future Research
We believe that more data introduced to the system (millions of instances) will lead to a slow system, hence deep learning methods could be researched and implemented into the system.







Reference
Ryan, M. (2019). Ethics of Using AI and Big Data in Agriculture: The Case of 
a Large Agriculture Multinational. ORBIT Journal, 2(2). 
https://doi.org/10.29297/orbit.v2i2.109
Ferris, Jody L, "Data Privacy and Protection in the Agriculture Industry: Is Federal Regu-
lation Necessary." Minn. JL Sci. & Tech, Vol. 18, Issue 1, 2017, pp. 309-342.
https://www.worldbank.org. 2022. Agriculture and Food. [online] Available at: <https://www.worldbank.org/en/topic/agriculture/overview#1> [Accessed 25 April 2022].
Morgan, J. B. & Connolly, E. L. (2013) Plant- Soil Interactions: Nutrient Uptake. Nature Education Knowledge 4(8):2
Nature.com. 2022. Plant-Soil Interactions: Nutrient Uptake | Learn Science at Scitable. [online] Available at: https://www.nature.com/scitable/knowledge/library/plant-soil-interactions-nutrient-uptake- 105289112/ [Accessed 25 April 2022].
Zhang, N., Wang, M. and Wang, N., 2022. Precision agriculture—a worldwide overview.






