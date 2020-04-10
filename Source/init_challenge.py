import numpy as np
from numpy import exp, linspace, random
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import itertools


def data_generator(n, data, al, be, ga, de):
    fx = al + be * data + ga * (data ** 2) + de * (data ** 3)

    plt.subplot(1, 2, 1)
    plt.plot(data, fx, 'r.')
    plt.ylabel('f(x)')
    plt.xlabel('x')

    plt.subplot(1, 2, 2)
    plt.plot(fx, 'bx')
    plt.xlabel('n')

    plt.show()
    return fx


def function_to_fit(x, a, b, c, d):
    return a + b * x + c * (x ** 2) + d * (x ** 3)


def nlin_fitting(x, y):
    init_vals = [1.0, 1.0, 1.0, 1.0]
    best_vals, covar = curve_fit(function_to_fit, x, y, p0=init_vals)
    print('Best parameters: {}'.format(best_vals))
    y_hat = function_to_fit(x, *best_vals)
    print('y_hat : \n', y_hat, '\n')
    plt.plot(x, y, 'b|', label='data')
    plt.plot(x, function_to_fit(x, *init_vals), 'r--',
             label='initial fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(init_vals))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.show()
    return best_vals, y_hat


def make_mutative_array(init_, s):
    arr = []
    for i in range(len(s)):
        if i == 0:
            arr.append(init_ + s[i])
        else:
            arr.append(s[i-1] + s[i])
    return np.array(arr)


def normpdf(bins, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2))


def plot_histogram_subplots(mutated_parameters, mu, sigma, title_str):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    count_a, bins_a, ignored_a = ax1.hist(mutated_parameters[0], 30, density=True)
    ax1.plot(bins_a, normpdf(bins_a, mu, sigma), linewidth=2, color='r')
    ax1.title.set_text('Histogram of ' + title_str[0])

    count_b, bins_b, ignored_b = ax2.hist(mutated_parameters[1], 30, density=True)
    ax2.plot(bins_b, normpdf(bins_b, mu, sigma), linewidth=2, color='r')
    ax2.title.set_text('Histogram of ' + title_str[1])
    plt.tight_layout()
    plt.show()


def gaussian_noise():
    params = [0.0, 1.0]
    mu, sigma = 0.0, 0.1
    s = np.random.normal(mu, sigma, 100)

    mutated_parameters = np.array([make_mutative_array(params[i], s) for i in range(len(params))])
    plot_histogram_subplots(mutated_parameters, mu, sigma, ['alpha', 'beta'])


def population_gen():
    mu, sigma = 0.0, 1.0
    parameters = [np.random.normal(mu, sigma, 100) for i in range(4)]
    population = np.array(parameters).T
    plot_histogram_subplots(population[:, [1, 2]].T, mu, sigma, ['beta', 'gamma'])
    # print(population)
    return population


def evolution(pop_, x, y):
    error_map = {}
    for idx, row in enumerate(pop_):
        rmse = np.sqrt(np.mean((y - function_to_fit(x, *row)) ** 2))
        error_map[idx] = rmse
    sorted_err_map = {k: v for k, v in sorted(error_map.items(), key=lambda item: item[1])}
    truncated_err_map = dict(itertools.islice(sorted_err_map.items(), 10))
    print(truncated_err_map)
    reduced_best_population = np.array([pop_[key] for key in truncated_err_map.keys()])
    new_pop = reduced_best_population.copy()
    mu, sigma = 0.0, 0.1
    s = np.random.normal(mu, sigma, 9)
    for idx, row in enumerate(reduced_best_population):
        new_pop = np.vstack((new_pop, np.array([make_mutative_array(row[i], s) for i in range(len(row))]).T))
    return new_pop, sorted_err_map


def main(given_param, x_data_lim, num_of_x, epochs):

    # Task 1
    num = random.randint(num_of_x[0], num_of_x[1])
    # x_data = random.uniform(-1, 1, num)
    x_data = linspace(x_data_lim[0], x_data_lim[1], num)
    y_data = data_generator(num, x_data, *given_param)

    # Task 2
    best_params, y_h = nlin_fitting(x_data, y_data)

    # Task 3
    gaussian_noise()

    # Task 4
    the_population = population_gen()

    # Task 5
    evolved_population, sorted_error_dict = evolution(the_population, x_data, y_data)

    # Task 6
    error_dict_list = [sorted_error_dict.copy()]
    for i in range(epochs-1):
        evolved_population, err_map = evolution(evolved_population, x_data, y_data)
        error_dict_list.append(err_map)
    evolution_wise_dict = {}
    for k in sorted_error_dict.keys():
        evolution_wise_dict[k] = list(evolution_wise_dict[k] for evolution_wise_dict in error_dict_list)
    print(evolution_wise_dict, len(evolution_wise_dict))

    y_axis = range(epochs)
    for key in evolution_wise_dict.keys():
        plt.plot(evolution_wise_dict[key], y_axis)
    plt.gca().invert_yaxis()
    plt.xlabel('RMSE error')
    plt.ylabel('Epochs')
    plt.title('Change in each gene RMSE over evolving population (epochs)')
    plt.show()


if __name__ == "__main__":
    main()
