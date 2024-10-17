#include <time.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../pbplots/pbPlots.h"
#include "../pbplots/supportLib.h"

#define INPUT_LAYER_SIZE 2
#define N_WEIGHTS (INPUT_LAYER_SIZE * 2)
#define GRADIENT_LAMBDA 0.001f
#define MIN_WEIGHT -1.f
#define MAX_WEIGHT  1.f
#define POINTS_DISTRIBUTION 2.f
#define EPSILON 1e-9f
#define ERROR_VALUES_MEMORY_LIMIT 1024
#define ERROR_TARGET EPSILON

typedef struct {
    double x, y;
} vec2;

typedef struct {
    vec2 point;
    double expected_output;
} testcase_t;

double sigmoid(double x)
{
    return 1.f / (1.f + pow(M_E, -x));
}

double sigmoid_derivative(double x, double x_derivative)
{
    double t = sigmoid(x);
    return t * (1 - t) * x_derivative;
}

double process_input(double *input_layer, double *weights)
{
    double x = 0;
    for (int i = 0; i < INPUT_LAYER_SIZE; i++) {
        x += weights[INPUT_LAYER_SIZE + i] * weights[i] * input_layer[i];
    }

    return sigmoid(x);
}

double calculate_gradient(double *input_layer, double *weights, double expected_output, double *result)
{
    double diff = process_input(input_layer, weights) - expected_output;

    double sum = 0;
    for (int i = 0; i < INPUT_LAYER_SIZE; i++)  {
        sum += weights[INPUT_LAYER_SIZE + i] * weights[i] * input_layer[i];
    }

    for (int i = 0; i < INPUT_LAYER_SIZE; i++) {
        result[i] = 2*diff * sigmoid_derivative(sum, input_layer[i] * weights[INPUT_LAYER_SIZE + i]);
    }

    for (int i = 0; i < INPUT_LAYER_SIZE; i++) {
        result[INPUT_LAYER_SIZE + i] = 2*diff * sigmoid_derivative(sum, input_layer[i] * weights[i]);
    }

    return diff*diff;
}

double update_weights(double *input_layer, double *weights, double expected_output)
{
    static double gradient[N_WEIGHTS];
    double error = calculate_gradient(input_layer, weights, expected_output, gradient);

    for (int i = 0; i < N_WEIGHTS; i++) {
        weights[i] -= GRADIENT_LAMBDA * gradient[i];
    }

    return error;
}

void shuffle(testcase_t *array, size_t n)
{
    if (n > 1) {
        for (size_t i = 0; i < n - 1; i++) {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          testcase_t tmp = array[j];
          array[j] = array[i];
          array[i] = tmp;
        }
    }
}

double random_double(double min, double max)
{
    return (max - min) * ((double)rand() / RAND_MAX) + min;
}

size_t dataset_size = 10000;
size_t n_epochs = 100;
size_t n_tests = 100;
int    auto_flag = 0;

void read_args(int argc, char **argv)
{
    int c;
    int digit_optind = 0;

    while (1) {
        static struct option long_options[] = {
            { "epochs", 1, NULL, 'e' },
            { "dataset", 1, NULL, 'd' },
            { "seed", 1, NULL, 's' },
            { "test", 1, NULL, 't' },
            { "auto", 0, NULL, 'a' },
            { NULL, 0, NULL, 0 }
        };

        c = getopt_long(argc, argv, "e:d:s:t:a", long_options, NULL);
        
        if (c == -1)
            break;

        switch (c) {
        case 'e':
            n_epochs = atoi(optarg);
            break;
        case 'd':
            dataset_size = atoi(optarg);
            break;
        case 's':
            srand(atoi(optarg));
            break;
        case 't':
            n_tests = atoi(optarg);
            break;
        case 'a':
            auto_flag = 1;
            break;
        default:
            exit(1);
        }
    }
}

int main(int argc, char **argv)
{
    srand(time(NULL));
    read_args(argc, argv);

    size_t n_error_values = n_epochs * dataset_size;
    size_t error_values_step = 1;
    if (n_error_values / 1024 / 1024 > ERROR_VALUES_MEMORY_LIMIT / sizeof(double)) {
        n_error_values = (ERROR_VALUES_MEMORY_LIMIT / sizeof(double)) * 1024 * 1024;
        error_values_step = (n_epochs * dataset_size + error_values_step - 1) / error_values_step;
    }

    testcase_t *dataset = malloc(sizeof(testcase_t) * dataset_size);
    double *error_values = malloc(sizeof(double) * n_error_values);

    if (dataset == NULL || error_values == NULL) {
        fprintf(stderr, "failed to allocate memory\n");
        return 1;
    }

    for (size_t i = 0; i < dataset_size; i++) {
        dataset[i].point.x = random_double(-POINTS_DISTRIBUTION, POINTS_DISTRIBUTION);
        dataset[i].point.y = random_double(-POINTS_DISTRIBUTION, POINTS_DISTRIBUTION);
        dataset[i].expected_output = (dataset[i].point.y > -dataset[i].point.x) ? 1.f : 0.f;
    }

    static double weights[N_WEIGHTS];
    for (size_t i = 0; i < N_WEIGHTS; i++) {
        weights[i] = random_double(MIN_WEIGHT, MAX_WEIGHT);
    }

    for (size_t epoch = 0, i = 0; epoch < n_epochs; epoch++) {
        printf("epoch %zu\n", epoch);
        shuffle(dataset, dataset_size);

        double error_sum;
        for (int j = 0; j < dataset_size; j++, i++) {
            double error = update_weights((double*)&dataset[j].point, weights, dataset[j].expected_output);
            error_sum += error;

            if ((i % error_values_step) == 0)
                error_values[i / error_values_step] = error;
        }

        if (error_sum / dataset_size < ERROR_TARGET) {
            n_epochs = epoch;
            n_error_values = i / error_values_step;
            break;
        }
    }

    size_t correct_cnt = 0;
    size_t wrong_cnt = 0;

    for (size_t i = 0; i < n_tests; i++) {
        vec2 input;
        double expected_output;

        input.x = random_double(-POINTS_DISTRIBUTION, POINTS_DISTRIBUTION);

        if (i % 2) { // correct
            input.y = -input.x + random_double(EPSILON, POINTS_DISTRIBUTION);
            expected_output = 1.f;
        }
        else { // wrong
            input.y = -input.x - random_double(EPSILON, POINTS_DISTRIBUTION);
            expected_output = 0.f;
        }

        double output = process_input((double*)&input, weights);
        double closest_output = (output < 0.5f) ? 0.f : 1.f;

        if (closest_output == expected_output)
            correct_cnt++;
        else
            wrong_cnt++;

        printf("%s: (%f, %f) --> %f\n", (closest_output == expected_output ? "OK" : "WRONG!!!!!"), input.x, input.y, output);
    }

    printf(
        "correct tests: %zu (%f%%)\n"
        "wrong tests: %zu (%f%%)\n",
        correct_cnt, (float)correct_cnt / n_tests * 100,
        wrong_cnt, (float)wrong_cnt / n_tests * 100
    );

    // plotting here

    StartArenaAllocator();
    system("mkdir -p plots");
    for (size_t n = n_error_values, i = 1; n > 1; n /= 2, i++) {
        static char plot_name[128];
        sprintf(plot_name, "plots/plot%zu.png", i);

        RGBABitmapImageReference *canvas = CreateRGBABitmapImageReference();
        int success = DrawBarPlot(canvas, 600, 400, error_values, n, NULL);

        ByteArray *pngdata = ConvertToPNG(canvas->image);
        WriteToFile(pngdata, plot_name);
        DeleteImage(canvas->image);

        printf("saved %s of size %zu\n", plot_name, n);

        for (int i = 0; i < n; i += 2) {
            error_values[i/2] = error_values[i];
        }
    }
    FreeAllocations();

    free(dataset);
    free(error_values);
}
