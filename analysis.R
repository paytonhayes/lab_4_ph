#Bring packages from library 
library(glmmTMB)
library(dplyr)
library(tidyr)
library(ggplot2)
library(broom.mixed)
library(performance)
library(gt)
library(readr)
library(DHARMa)
library(ggeffects)
library(titanic)


titanic <- titanic_train

#Keep the same complete-case sample for every model

titanic <- titanic |>
  select(Survived, Sex, Age, Pclass) |>
  drop_na() |>
  mutate(
    Sex.Binary = case_when(
      Sex == "Male" ~ 0,
      Sex == "Female" ~ 1)) |>
  mutate(
    Survived.Char = case_when(
      Survived == 0 ~ "Died",
      Survived == 1 ~ "Survived")) |>
  mutate(
    Pclass.Char = case_when(
      Pclass == "1" ~ "First Class",
      Pclass == "2" ~ "Second Class",
      Pclass == "3" ~ "Third Class")) |>
  mutate(Pclass = factor(Pclass, levels = c(1, 2, 3))) |>
  select(Survived, Survived.Char, Sex, Sex.Binary, Pclass, Pclass.Char, Age)

n <- nrow(titanic)

#Explore Data in console using: head(), str(), summarise()

#Part 1.4 

#A formatted table reporting the mean, standard deviation, minimum, 
#and maximum for the continuous variable (age). Add title and bold using md()


age_summary <- titanic |>
  summarise(
    Mean = mean(Age),
    SD = sd(Age),
    Min = min(Age),
    Max = max(Age)
  )
age_summary |>
  gt() |>
  tab_header(
    title = md("**Titanic Passenger Ages**")
  )|>
  tab_options(
    table.align = "left"
  )|>
  tab_options(
    table.width = pct(75)
  )

#A formatted table reporting the cell counts for each categorical variable.

#survival

survivalcounts <- titanic |>
  count(Survived.Char)

survivalcounts |>
  gt() |>
  tab_header(
    title = md("**Passenger Survival Counts**")
  ) |>
  cols_label(
    Survived.Char = "",
    n = "")|>
  tab_options(
    table.align = "left"
  )|>
  tab_options(
    table.width = pct(75)
  )


#sex table

sexcounts <- titanic |>
  count(Sex)|>
  mutate(Sex = stringr::str_to_title(Sex))

sexcounts |>
  gt() |>
  tab_header(
    title = md("**Number of Passengers by Sex**")
  ) |>
  cols_label(
    Sex = "",
    n = "" )|>
  tab_options(
    table.align = "left"
  )|>
  tab_options(
    table.width = pct(75)
  )

# passenger class (with a re name of the classes)

classcounts <- titanic |>
  count(Pclass) |>
  mutate(
    Pclass = recode(
      as.character(Pclass),
      "1" = "First Class",
      "2" = "Second Class",
      "3" = "Third Class"
    )
  )

classcounts |>
  gt() |>
  tab_header(
    title = md("**Number of Passengers by Class**")
  ) |>
  cols_label(
    Pclass = "",
    n = ""
  )|>
  tab_options(
    table.align = "left"
  )|>
  tab_options(
    table.width = pct(75)
  )

#A short paragraph describing the most important patterns you observe.

#In terms of the passengers on the Titanic, the average age was 29.7 years. 
#The age dispersion was quite high, with the youngest being just 0.42 years, or about 5 months old, and the oldest passenger being 80 years old. 
#More people died than survived, being 424 versus 290 passengers respectively. 
#The spread between men and women was not equal, with there being 1.74 times more men on board the ship. 
#Third class passengers represented almost half of the ship (49.72%), while first and second class represented 26.05% and 24.23% respectively. 

## **Part 2: Competing Binary Logistic Regression Models**

#I believe that, at the time, the common societal norm of "women and children first" would have played a very large role in the survival rates of both female passengers and children. 
#Because of this, I would say that sex and age will both be strong factors in our prediction model. Social economic status (ticket class) would also provide a good predictor of survival. 
#First class passengers would have probably been given priority to lifeboats and have been closer to the upper deck to board them. In contrast, third class passengers might have been lower down on the ship, 
#taken longer to get to the boats, and might have not had the opportunity to board them at all. Looking at a potential model, I predict that age, sex, and class status will predict survival rates, with class and sex being the strongest predictors.

#### **2.2 Nested Models**

#Use glmmTMG() to estimate the models and then assign each of them a name

# Intercept-only mode, Null

mod0 <- glmmTMB(
  formula = Survived ~ 1,
  data = titanic, 
  family = binomial(link = "logit")
)

# Sex model

mod1 <- glmmTMB(
  formula = Survived ~ Sex,
  data = titanic, 
  family = binomial(link = "logit")
)

# Sex + Pclass model

mod2 <- glmmTMB(
  formula = Survived ~ Sex + Pclass,
  data = titanic, 
  family = binomial(link = "logit")
)

# Expanded model: Sex + Pclass + Age

mod3 <- glmmTMB(
  formula = Survived ~ Sex + Pclass + Age,
  data = titanic, 
  family = binomial(link = "logit")
)

#In the R Console, use summary() to inspect the results of each fitted model


# **Part 3: Comparing Model Performance**

# 3.1: Conduct nested-model χ²-tests to test all mods at the same time (anova)

tests <- anova(mod0, mod1, mod2, mod3)

#select form tests: Residual degrees of freedom, Change in degrees of freedom, 
#χ²-statistic for the nested-model comparison, p-value for the #nested-model comparison.

tests <- tests |>
  select(Df, Chisq, `Chi Df`, `Pr(>Chisq)`) |>
  rename(
    Residual.df = Df,
    Nested.df = `Chi Df`,
    Nested.Chisq.stat = Chisq,
    Nested.p.value = `Pr(>Chisq)`)

## **3.2. Extracting Nested Model-Level Performance Statistics**

#use broom.mixed::glance() to extract nested model-level statistics for each of the four fitted models.

glance0 <- broom.mixed::glance(mod0)
glance1 <- broom.mixed::glance(mod1)
glance2 <- broom.mixed::glance(mod2)
glance3 <- broom.mixed::glance(mod3)

#combine the glance results into one data frame (glances) using rbind() or bind_rows()

#store vector 
models <- c(
  "mod0: Null",
  "mod1: Sex",
  "mod2: Sex + Pclass",
  "mod3: Sex + Pclass + Age")

glances <- rbind(
  glance0,
  glance1,
  glance2,
  glance3)

#add in model name column to glances

glances <- glances |>
  mutate(
    Model = c(
      "mod0: Null",
      "mod1: Sex",
      "mod2: Sex + Pclass",
      "mod3: Sex + Pclass + Age"
    ),
    .before = 1
  )

# 3.3. Combine nested tests and model performance using cbind()

mod_comparison_table <- cbind(
  glances,
  tests)

#select and assign it to itself: model, AIC, BIC< loglink, deviance, df.res, nest.df, nest.chi, nestp

mod_comparison_table <- mod_comparison_table |>
  select(
    Model,
    AIC,
    BIC,
    logLik,
    deviance,
    df.residual,
    Nested.df,
    Nested.Chisq.stat,
    Nested.p.value
  )

#format gt table: 

mod_comparison_table_gt <- mod_comparison_table|> 
  gt() |>
  cols_label(
    Model = "Model",
    AIC = "AIC",
    BIC = "BIC",
    logLik = "Log-likelihood",
    deviance = "Deviance",
    df.residual = "Residual df",
    Nested.df = "Nested df",
    Nested.Chisq.stat = "Nested χ²",
    Nested.p.value = "Nested p"
  ) |>
  fmt_number(
    columns = c(
      AIC,
      BIC,
      logLik,
      deviance,
      Nested.Chisq.stat
    ),
    decimals = 3
  ) |>
  fmt(
    columns = Nested.p.value,
    fns = function(x) {
      ifelse(
        is.na(x),
        "—",
        ifelse(x < .001, "< .001", sprintf("%.3f", x))
      )
    }
  ) |>
  sub_missing(
    columns = c(Nested.df, Nested.Chisq.stat),
    missing_text = "—"
  )

# compare preformance and save

performance_comparison <- compare_performance(
  mod0,
  mod1,
  mod2,
  mod3,
  rank = TRUE)

performance_comparison
plot(performance_comparison)

selected_row <- mod_comparison_table |>
  filter(Nested.p.value < .05) |>
  slice_tail(n = 1)

selected_name <- selected_row$Model



## **Part 4: Evaluating Binary Logistic Regression Diagnostics**

### **4.1 Binary Logistic Regression Diagnostics**


#DHARMa:: diagnostic plots for Linearity to the logit scale, Collinearity, Complete separation, Influential observations, Events per predictor

##DHARMa:: diagnostic plots for Linearity to the logit scale
set.seed(505)

res_mod3 <- simulateResiduals(
  mod3, 
  n = 500, 
  plot = FALSE)

plot(res_mod3)

#colinerairty and save it in a table
#colinerarity is good

collinearity <- check_collinearity(mod3)

#Influential observations

outliers_mod3 <- testOutliers(res_mod3)

#dispersion

dispersion_mod3 <- testDispersion(res_mod3)


### **4.2: Diagnostic Interpretations** 

#Avoid treating binary logistic regression assumptions as a mechanical checklist. Explain what the diagnostic results imply for the interpretation of your findings.

## **Part 5: Preferred Model**

#Write a paragraph explaining why your preferred model is more appropriate than the alternatives.

### **5.2. Coefficient table**

#use tidy::broom.mixed to extract coefficient-level results from a fitted binary logistic regression model.

coefficient_results <- broom.mixed::tidy(
  mod3,
  effects = "fixed",
  component = "cond",
  conf.int = TRUE,
  conf.level = 0.95)

# create binary logistic regression coefficient gt table 

coefficient_results |>
  select(
    term,
    estimate,
    std.error,
    conf.low,
    conf.high,
    p.value
  ) |>
  gt() |>
  tab_header(
    title = md("**mod3 Binary Logistic Regression Coefficients**")
  ) |>
  cols_label(
    term = "Predictor",
    estimate = "Estimate",
    std.error = "SE",
    conf.low = "95% CI lower",
    conf.high = "95% CI upper",
    p.value = "p"
  ) |>
  fmt_number(
    columns = c(
      estimate,
      std.error,
      conf.low,
      conf.high
    ),
    decimals = 3
  ) |>
  fmt(
    columns = p.value,
    fns = function(x) {
      ifelse(
        x < .001,
        "< .001",
        sprintf("%.3f", x)
      )
    }
  )


### **5.3. Interpreting Coefficients**

# Coefficients are on the log-odds scale, make them odds ratio and CI. Save onto themselves for each

log.co <- fixef(mod3)$cond

expo.co <- exp(fixef(mod3)$cond)

or_ci <- exp(confint(mod3, parm = "beta_"))

### **5.4. Overall Model Performance**

#Using gt(), create a table reporting the overall model performance for your preferred model.

selected_row |>
  select(
    Model,
    AIC,
    BIC,
    logLik,
    deviance,
    df.residual,
    Nested.df,
    Nested.Chisq.stat,
    Nested.p.value
  ) |>
  gt() |>
  cols_label(
    Model = "Model",
    AIC = "AIC",
    BIC = "BIC",
    logLik = "Log-likelihood",
    deviance = "Deviance",
    df.residual = "Residual df",
    Nested.df = "Nested df",
    Nested.Chisq.stat = "Nested χ²",
    Nested.p.value = "Nested p"
  ) |>
  fmt_number(
    columns = where(is.numeric),
    decimals = 2
  )

### **5.5. Summarizing Results**

#Provide a summary of the analysis you conducted, along with the results of your preferred model (in the metric of odd ratios).


### **5.6. Visualizing Age and Survival** 

#Using ggeffects::ggpredict() and ggplot2, select one predictor from your preferred model and visualize it’s association with the outcome variable, Survived.

#age and survival, 

pred_age <- ggpredict(
  mod3, 
  terms = "Age")

ggplot(pred_age, aes(x, predicted)) +
  geom_ribbon(
    aes(
      ymin = conf.low, 
      ymax = conf.high), 
    alpha = 0.18) +
  geom_line(
    linewidth = 1.2) +
  labs(
    title = "Predicted Probability of Survival by Passenger Age",
    x = "Age of Passenger",
    y = "Predicted Probability of Survival"
  ) +
  theme_classic()+
  theme(
    plot.title = element_text(
      face = "bold",
      hjust = 0.5
    ),
    axis.title.y = element_text(
      face = "bold",
      margin = margin(r = 20)
    ),
    axis.title.x = element_text(
      face = "bold",
      margin = margin(t = 20)
    )
  )

## **Part 6: Limitations and Discussions**
# What do your results suggest about the relative importance of sex, ticket type (proxy for SES), and age for survival outcomes on the Titanic?
  
