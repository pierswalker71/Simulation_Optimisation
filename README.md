# Simulation Optimisation

Using machine learning and optimisation to find low-cost input settings for a simulated business process.

## Overview

This project demonstrates how a simulation model can be combined with machine learning and optimisation to support better operational decision-making.

Many organisations use simulation tools to model complex processes, such as manufacturing output, service capacity, staffing demand, resource allocation or operational throughput. These simulators are useful, but they can be difficult to optimise manually. Running every possible combination of inputs is often slow, expensive or impractical.

This notebook shows a practical alternative:

1. generate simulation results from a range of input values
2. train a machine learning model to approximate the simulator
3. use optimisation to search for low-cost input combinations
4. validate the recommended settings against the original simulator

The result is a prototype decision-support workflow that identifies input configurations likely to achieve a target output at the lowest cost.

---

## Business Problem

A business has a simulation model that estimates output from a set of operational inputs.

For example, the inputs could represent:

* machines
* production lines
* staff groups
* depots
* service teams
* processing units
* capacity in different locations

The business wants to achieve a specific output target while keeping costs as low as possible.

In this demonstration, the target is to achieve an output of **60 units** using five integer input variables. Each input has an associated cost, so the best result is not simply the largest set of inputs. The best result is the lowest-cost configuration that achieves the target within an acceptable tolerance.

The core question is:

> What combination of inputs achieves the required output at the lowest cost?

This is a common operational problem. Increasing resources may improve output, but it also increases cost. The relationship between inputs and outputs may also be complex, non-linear and difficult to optimise by inspection.

---

## Why This Matters

Traditional simulation helps answer:

> What happens if we try this scenario?

Simulation optimisation goes further and helps answer:

> Which scenario should we choose?

This project demonstrates how an organisation could move from manually testing individual scenarios to automatically searching for better options.

The approach is valuable where:

* the simulator is expensive or slow to run
* there are many possible input combinations
* the relationship between inputs and outputs is not obvious
* decision-makers need to balance performance and cost
* the organisation wants explainable candidate solutions rather than a black-box recommendation.

---

## What the Tool Does

The notebook implements an end-to-end simulation optimisation workflow.

### 1. Sets the target and cost assumptions

The notebook defines:

* the target output
* the number of input variables
* the cost of each input
* the acceptable tolerance around the target output

This frames the optimisation as a practical business decision rather than a purely technical modelling problem.

### 2. Generates simulation data

The notebook runs a simulated process using many different combinations of input values.

Each simulation run records:

* the input values
* the simulated output
* the total cost of the chosen inputs

This creates a dataset that can be used to learn how the simulated process behaves.

### 3. Trains regression models

The notebook uses machine learning regression models to learn the relationship between the input values and the simulator output.

The modelling workflow includes:

* train/test splitting
* feature scaling
* polynomial feature generation
* model pipelines
* model comparison
* hyperparameter tuning
* cross-validation
* performance evaluation on unseen data

The purpose of the model is not just prediction. It acts as a **surrogate model** for the simulator.

### 4. Optimises the surrogate model

Once the model has learned the simulator’s behaviour, the notebook uses optimisation to search for input values that are predicted to meet the target output at minimum cost.

This is faster and more systematic than manually testing scenarios.

### 5. Validates candidate solutions

The predicted optimum is passed back through the original simulator to check whether it performs as expected.

The notebook also explores nearby input combinations to identify other compliant low-cost solutions.

### 6. Reviews the best compliant results

The final outputs identify input combinations that:

* achieve the target output within the accepted tolerance
* minimise total cost
* provide practical candidate scenarios for review

---

## Example Workflow

```text
Define target and cost assumptions
        ↓
Generate simulation runs
        ↓
Train machine learning model
        ↓
Use model as a surrogate for the simulator
        ↓
Optimise input values
        ↓
Validate against simulator
        ↓
Review lowest-cost compliant solutions
```

---

## Screenshots

### Model Performance

<img src="/images/simulator_optimisation_regression_performance.png"
     alt="Map preview of the school allocation tool"
     width="360">

Shows how well the selected regression model predicts simulator outputs on unseen test data.

### Optimisation Result

```markdown
![Optimisation result](docs/images/optimisation-result.png)
```

Shows the recommended input configuration, predicted output and estimated cost.

### Best Compliant Results

```markdown
![Best compliant results](docs/images/best-compliant-results.png)
```

Shows the lowest-cost solutions that achieve the target output within tolerance.

### Local Search Around the Optimum

```markdown
![Local search results](docs/images/local-search-results.png)
```

Shows how nearby input combinations compare with the predicted optimum.

---

## Technical Approach

The notebook uses Python and common data science libraries to build the workflow.

Key techniques include:

* simulation data generation
* supervised regression modelling
* surrogate modelling
* scikit-learn pipelines
* train/test validation
* cross-validation
* hyperparameter search
* feature scaling
* polynomial feature engineering
* global optimisation
* cost-based objective functions
* validation of model-generated recommendations

The optimisation process uses the trained regression model to estimate simulator output and then searches for input combinations that minimise cost while achieving the target.

---

## Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* scikit-learn
* SciPy
* Matplotlib / visualisation tools

---

## Repository Contents

```text
20220106 Simulation Optimiser.ipynb
```

The notebook contains the full prototype workflow, from simulation setup through to model training, optimisation and validation.

Recommended future structure:

```text
.
├── README.md
├── 20220106 Simulation Optimiser.ipynb
├── docs/
   └── images/
       ├── model-performance.png
       ├── optimisation-result.png
       ├── best-compliant-results.png
       └── local-search-results.png

```

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/pierswalker71/Simulation_Optimisation.git
cd Simulation_Optimisation
```

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn scipy matplotlib jupyter
```

Open the notebook:

```bash
jupyter notebook
```

Then run:

```text
20220106 Simulation Optimiser.ipynb
```

---

## Value Delivered

This project shows how data science can support better operational decisions.

The main value is that it turns simulation from a passive testing tool into an active optimisation tool.

Instead of asking users to manually test many scenarios, the workflow helps identify promising low-cost configurations automatically.

Potential business benefits include:

* faster scenario analysis
* lower-cost resource planning
* better use of existing simulation models
* clearer trade-offs between cost and output
* more systematic decision-making
* reusable optimisation logic for other operational problems

---

## What This Demonstrates

This project demonstrates the ability to:

* translate a business planning problem into a data science workflow
* generate useful data from a simulated process
* train models to approximate complex input-output relationships
* evaluate model performance using appropriate validation methods
* use optimisation to recommend decisions
* incorporate cost into analytical decision-making
* validate model-generated recommendations against the original process
* communicate technical outputs as practical business options

The project is particularly relevant to:

* operational analytics
* decision-support tools
* simulation modelling
* applied machine learning
* optimisation
* AI-enabled planning tools

---

## Limitations

This is a prototype notebook rather than a production system.

Current limitations include:

* the simulator is synthetic rather than connected to a live business process
* the cost function is simplified
* the example uses a fixed target output
* the input space is limited to five integer variables
* uncertainty in model predictions is not explicitly quantified
* there is no user interface
* results require domain validation before use in real decisions
* the notebook has not yet been refactored into reusable Python modules

These limitations are appropriate for a prototype, but they indicate where the project could be developed further.

---

## Possible Extensions

Useful future improvements could include:

* refactoring the notebook into reusable Python modules
* adding a user interface for non-technical users
* allowing users to change targets, costs and constraints interactively
* supporting multiple objectives, such as cost, quality, resilience and service level
* adding uncertainty estimates around recommendations
* comparing different optimisation methods
* integrating with a real simulation engine
* exporting recommended scenarios to CSV or Excel
* adding automated reporting for decision-makers

---

## Portfolio Reflection

This project is a useful portfolio example because it shows applied data science beyond prediction alone.

The notebook demonstrates how modelling can be used to support decisions. It combines simulation, machine learning and optimisation to answer a practical business question: how to achieve a required output at the lowest cost.

The project also shows the value of surrogate modelling. Instead of relying on exhaustive simulation, the workflow learns from simulation runs and uses that learned model to guide the search for better options.

Although the notebook is a prototype, it demonstrates a complete analytical pattern that could be adapted to real-world operational planning problems.

The most important capability demonstrated is the ability to connect technical methods to business value:

* simulation generates evidence
* machine learning learns the system behaviour
* optimisation recommends action
* validation checks whether the recommendation works

That combination is directly relevant to building practical AI and analytics tools for decision support.

---

## Summary

This project demonstrates a simulation optimisation workflow in Python.

It shows how a business simulator can be treated as a black box, approximated using machine learning, and then optimised to find low-cost input configurations that achieve a target output.

The result is a practical prototype for cost-aware operational decision support.
