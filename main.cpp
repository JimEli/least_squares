// Demonstration of least squares linear, quardatic and polynomial regression.
// 11/9/2020 JME.
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>

struct Point { double x, y; };

template <typename T>
std::vector<T> operator- (std::vector<T> const& first, std::vector<T> const& second)
{
    std::vector<T> result;
    std::transform(first.begin(), first.end(), second.begin(), std::back_inserter(result), std::minus<T>());
    return result;
}

template <typename T>
std::vector<T> operator* (std::vector<T> const& first, const double factor)
{
    std::vector<T> result;
    std::transform(first.begin(), first.end(), std::back_inserter(result), [factor](double a) { return a * factor; });
    return result;
}

double T(const std::vector<Point>& points, const double n, const double m)
{
    return std::accumulate(begin(points), end(points), 0., [&](double result, Point p) -> double { return result + pow(p.x, n) * pow(p.y, m); });
}

std::vector<double> solveQuadratic(const std::vector<Point>& points)
{
    double a = 0., b = 0., c = 0., d = 0., e = 0., f = 0., g = 0.;
    size_t	n = points.size();

    for (size_t i = 0; i < n; i++)
    {
        double x2 = points[i].x * points[i].x;
        double x3 = x2 * points[i].x;
        double x4 = x2 * x2;
        a = a + x4;
        b = b + x3;
        c = c + x2;
        d = d + points[i].y * x2;
        e = e + points[i].x;
        f = f + points[i].y * points[i].x;
        g = g + points[i].y;
    }

    double c2be = c * c - b * e;
    double e2cn = e * e - c * n;
    double bdaf = b * d - a * f;
    double cfbg = c * f - b * g;
    double acb2 = a * c - b * b;
    double denom = (b * c - a * e) * c2be - (c * e - b * n) * (b * b - a * c);

    std::vector<double> coefficients;
    coefficients.push_back(((c * d - b * f) * e2cn - (e * f - c * g) * c2be) / (acb2 * e2cn + c2be * c2be));
    coefficients.push_back((cfbg * (b * c - a * e) - bdaf * (c * e - b * n)) / denom);
    coefficients.push_back((bdaf * (c * c - b * e) + cfbg * acb2) / denom);

    return coefficients;
}

std::vector<double> solveLinear(const std::vector<Point>& points)
{
    double	a = 0., b = 0., c = 0., d = 0.;
    size_t n = points.size();

    for (size_t i = 0; i < n; i++)
    {
        a = a + points[i].x * points[i].x;
        b = b + points[i].x;
        c = c + points[i].x * points[i].y;
        d = d + points[i].y;
    }

    std::vector<double> coefficients;
    // m and b.
    coefficients.push_back((b * d - c * n) / (b * b - a * n));
    coefficients.push_back((b * c - a * d) / (b * b - a * n));

    return coefficients;
}

double calcYValue(double const x, const std::vector<double>& c)
{
    size_t i = c.size() - 1;
    return std::accumulate(c.begin(), c.end(), 0., [&](double a, double b) mutable { return a + b * pow(x, i--); });
}

static void stats(const std::vector<Point>& pts, const std::vector<double>& coefficients)
{
    double mu = 0., sigma2 = 0., r = 0., ymu = 0.;
    double dividend, divisor;
    size_t n = pts.size();

    for (size_t i = 0; i < n; i++)
    {
        // mean.
        mu += fabs(pts[i].y - calcYValue(pts[i].x, coefficients));
        // y mean.
        ymu += pts[i].y;
    }
    mu /= n;
    ymu /= n;

    // variance.
    for (size_t i = 0; i < n; i++)
        sigma2 += pow((mu - (pts[i].y - calcYValue(pts[i].x, coefficients))), 2.0);
    sigma2 /= n;

    // standard deviation.
    sigma2 = sqrt(sigma2);

    // variance y prime.
    dividend = 0.;
    for (size_t i = 0; i < n; i++)
        dividend += pow(((calcYValue(pts[i].x, coefficients)) - ymu), 2.0);
    dividend /= n;

    // variance y.
    divisor = 0.;
    for (size_t i = 0; i < n; i++)
        divisor += pow((pts[i].y - ymu), 2.0);
    divisor /= n;

    // correlation coefficient.
    r = sqrt(dividend / divisor);

    std::cout << "mu = " << mu << "\nsigma2 = " << sigma2 << "\nr = " << r << "\n\n";
}

void displayMatrix(std::vector<std::vector<double>> matrix)
{
    std::cout << std::fixed << std::setprecision(4) << "\n\n";

    for (auto row : matrix)
    {
        std::cout << "|\t";
        for (auto val : row)
            std::cout << val << "\t";
        std::cout << "|\n";
    }
}

void displayCoefficients(const std::vector<double>& coefficients, const double degree)
{
    std::cout << "y = ";

    for (size_t i = 0; i < coefficients.size(); i++)
    {
        int power = static_cast<int>(degree - i);
        if (power > 1)
        {
            std::cout << coefficients[i];
            std::cout << "x^" << power << " + ";
        }
        else if (power == 1)
            std::cout << coefficients[i] << "x + ";
        else if (power == 0)
            std::cout << coefficients[i];
    }
    std::cout << "\n";
}

int main(int argc, char** argv)
{
    // argv[1] = data file name, argv[2] = degree.
    if (argc < 2)
    {
        std::cout << "Requires File With Points List\n";
        return -1;
    }

    // Polynomial degree.
    double degree = 4;
    if (argc > 2)
        degree = std::stoi(std::string(argv[2]));

    // Attempt to read data file.
    std::vector<Point> points;
    int length;
    std::ifstream file(argv[1]);
    file >> length;
    while (length--)
    {
        int n;
        double x, y;
        
        file >> n >> x >> y;
        points.push_back(Point{ x, y });
    }
    file.close();

    // Perform polynomial fit.
    {
        // Create square matrix of polynomial degree.
        std::vector<std::vector<double>> matrix;
        for (int row_index = 0; row_index <= degree; row_index++)
        {
            std::vector<double> row;

            // Sum of the squares of the x terms.
            for (int col_index = 0; col_index <= degree; col_index++)
                row.push_back(T(points, 2 * degree - row_index - col_index, 0));

            row.push_back(T(points, degree - row_index, 1));
            matrix.push_back(row);
        }

        // Gauss-Jordan elimination.
        int n = 0;
        for (size_t i = 0; i < matrix.size(); i++)
        {
            matrix[i] = (matrix[i] * (1.0 / matrix[i][n]));
            for (size_t j = i + 1; j < matrix.size(); j++)
                matrix[j] = matrix[j] - (matrix[i] * matrix[j][n]);
            n++;
        }
        for (size_t i = 0; i <= matrix.size() - 2; i++)
        {
            int index_row = matrix.size() - 2 - i;
            for (size_t j = 0; j <= i; j++)
            {
                int reference_row = matrix.size() - 1 - j;
                int reference_index = matrix[0].size() - 2 - j;
                matrix[index_row] = matrix[index_row] - (matrix[reference_row] * matrix[index_row][reference_index]);
            }
        }

        // Extract the polynomial coefficients.
        std::vector<double> coefficients;
        for (size_t i = 0; i < matrix.size(); i++)
            coefficients.push_back(matrix[i][matrix[i].size() - 1]);

        displayCoefficients(coefficients, degree);
        stats(points, coefficients);
    }

    // Perform a quadratic fit.
    {
        std::vector<double> coefficients = solveQuadratic(points);
        displayCoefficients(coefficients, 2);
        stats(points, coefficients);
    }

    // Linear least squares.
    {
        std::vector<double> coefficients = solveLinear(points);
        displayCoefficients(coefficients, 1);
        stats(points, coefficients);
    }

    return 0;
}
