## Evolutionary Algorithm
### Test Task
 Write own evolutionary algorithm to solve an optimization problem.
 
 **1.) Data  Generator**:
 
 -  Write a piece of code which produces our ground truth data
as y=f(x) where f (x)=−0.1 + 0.3x – 0.7x^2 + 0.1x^3 and x should be in the range [-
1,1]. Produce a plot of f(x).

**2.) Fit Function**:

- The function we’d like to fit is f(x ; α, β, γ, δ) = α + βx + γx^2 + δx^3. Write a piece of code which produces y_hat = f(x ; α, β, γ, δ). Produce a plot for all parameters equal to one.

**3.) Mutation of Genes**:

-  A Gene in our algorithm is the set of parameters α, β, γ, δ, for f(x). Write a piece of code which randomly adds Gaussian noise to a set of parameters. Produce a histogram of α and one of β with 100 successive mutations each. Use as starting point α = 0.0 and β = 1.0. The Gaussian noise should have
zero mean and a standard deviation of 0.1.

**4.) Produce a Population**:

-  A population is a set of genes. Write a piece of code which randomly produces a population of size 100. The parameters should be Gauss distributed with mean zero and standard deviation one. Plot a histogram for β and γ.

**5.) Evolving a Population**:

- Evolution is performed by producing a new generation of genes
	-***a.) Evaluate Genes***:  For each gene compute y_hat and compute the loss using RMSE with respect to y.
	-***b.) Keep best***: Sort the genes by loss and keep the best 10
	-***c.) Mutate***: Mutate the 10 best genes until you have a total of 100 again.
	
**6.) Fit the Function**:

- Keep evolving your population for some time and monitor the loss of the best 10 genes as function of generations.

## Running the program

Run the Source/run_this.py file.
The parameter passed can be changed in this file.

## Results

The output graphs for all tasks and the console output are stored in the Graphs_and_output/ folder

### Example: Population Evolution over multiple epochs and gene mutations

<div>
<img src="/Graphs_and_output/Task5_task6_population_evolution.png">
</div>