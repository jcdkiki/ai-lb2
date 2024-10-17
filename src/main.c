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

int main(int argc, char **argv)
{
    argc--; argv++;
    if (argc != 4) {
        fprintf(stderr, "usage: lb2 SEED N_EPOCHS DATASET_SIZE N_TESTS\n"
                        "example: lb2 200 100 10000 100\n");
        return 1;
    }

    srand(atoi(argv[0]));
    size_t n_epochs = atoi(argv[1]);
    size_t dataset_size = atoi(argv[2]);
    size_t n_tests = atoi(argv[3]);

    size_t n_error_values = n_epochs * dataset_size;
    size_t error_values_step = 1;
    if (n_error_values / 1024 / 1024 > ERROR_VALUES_MEMORY_LIMIT / sizeof(double)) {
        n_error_values = (ERROR_VALUES_MEMORY_LIMIT / sizeof(double)) * 1024 * 1024;
        error_values_step = (n_epochs * dataset_size + error_values_step - 1) / error_values_step;
    }

    testcase_t *dataset = malloc(sizeof(testcase_t) * dataset_size);
    double *error_values = malloc(sizeof(double) * n_error_values);
    double *error_values_xaxis = malloc(sizeof(double) * n_error_values);

    if (dataset == NULL || error_values == NULL || error_values_xaxis == NULL) {
        fprintf(stderr, "failed to allocate memory\n");
        return 1;
    }

    for (size_t i = 0; i < n_error_values; i++) {
        error_values_xaxis[i] = i * error_values_step;
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
        printf("epoch %zu/%zu\n", epoch, n_epochs);
        shuffle(dataset, dataset_size);

        for (int j = 0; j < dataset_size; j++, i++) {
            double error = update_weights((double*)&dataset[j].point, weights, dataset[j].expected_output);

            if ((i % error_values_step) == 0)
                error_values[i / error_values_step] = error;
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

    // plotting

    StartArenaAllocator();
    system("rm -rf plots");
    system("mkdir -p plots");
    for (size_t n = n_error_values, i = 1; n > 2; n /= 2, i++) {
        static char plot_name[128];
        sprintf(plot_name, "plots/plot%zu.png", i);

        RGBABitmapImageReference *canvas = CreateRGBABitmapImageReference();
        int success = DrawScatterPlot(canvas, 600, 400, error_values_xaxis, n, error_values, n, NULL);

        ByteArray *pngdata = ConvertToPNG(canvas->image);
        WriteToFile(pngdata, plot_name);
        DeleteImage(canvas->image);

        printf("saved %s of size %zu\n", plot_name, n);

        for (int i = 0; i < n; i += 2) {
            error_values[i/2] = (error_values[i] + error_values[i+1]) / 2;
            error_values_xaxis[i/2] = error_values_xaxis[i+1];
        }
    }
    FreeAllocations();

    system("ffmpeg -y -framerate 2 -i 'plots/plot%d.png' plots/plot-2fps.mp4 2> /dev/null");
    system("ffmpeg -y -framerate 30 -i 'plots/plot%d.png' plots/plot-30fps.mp4 2> /dev/null");
    system("ffmpeg -y -framerate 2 -i 'plots/plot%d.png' plots/plot-2fps.gif 2> /dev/null");
    system("ffmpeg -y -framerate 30 -i 'plots/plot%d.png' plots/plot-30fps.gif 2> /dev/null");

    free(dataset);
    free(error_values);
    free(error_values_xaxis);
}
