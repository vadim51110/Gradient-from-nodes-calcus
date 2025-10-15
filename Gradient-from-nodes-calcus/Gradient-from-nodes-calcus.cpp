#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <cstdio>
#include "nanoflann.hpp"
#pragma warning(disable:4996)
const size_t n_inter = 16;
const double L = 4e2;
const size_t n_group = 9;
const size_t LSM_iter_phi = 10;
const size_t LSM_iter_param_x = 100;
const size_t LSM_iter_param_y = 10;
const size_t LSM_iter_param_z = 10;
const size_t Nstep = 600;

struct Adapter {
    double** points;
    size_t n;

    Adapter(double** pts, size_t count) : points(pts), n(count) {}

    size_t kdtree_get_point_count() const { return n; }

    double kdtree_get_pt(size_t idx, size_t dim) const {
        return points[idx][dim];
    }

    template<class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};


typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, Adapter>,
    Adapter, 3, size_t> Mytree;

static int re_ind(int i, size_t n)
{
    while (i >= n)
    {
        i -= n;
    }
    while (i < 0)
    {
        i += n;
    }
    return i;

}

static double det3_3(double** a, size_t n)
{
    int i, j;
    double res = 0;
    double mul_1;
    double mul_2;
    for (i = 0; i < n; i++)
    {
        mul_1 = 1;
        mul_2 = 1;
        for (j = 0; j < n; j++)
        {
            mul_1 = mul_1 * a[j][re_ind(j + i, n)];
            mul_2 = mul_2 * a[j][re_ind(-j + i, n)];
        }
        res += mul_1 - mul_2;
    }
    return res;
}

static void Transposition(double* res, double* a, size_t n, size_t m)
{
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++)
            res[j * n + i] = a[i * m + j];
}

static void multiplier(double* res, double* a, double* b, size_t n, size_t sum_dim, size_t m)
{
    int i, j, k;
    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++)
            res[i * m + j] = 0;

    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++)
            for (k = 0; k < sum_dim; k++)
            {
                res[i * m + j] += a[i * sum_dim + k] * b[k * m + j];
            }
}


static double LSMcalcus(double** coordif, double Ex, double Ey, double Ez, double phi0)
{
    int i;
    int fullnes;
    int prew_fullnes;
    double delta_phi_sq[n_group * 4];
    double delta_phi_sq_E;
    double delta_phi_sq_E_iter;
    delta_phi_sq_E = 0;
    delta_phi_sq_E_iter = 0;
    for (i = 0; i < n_group * 4; i++)
    {
        delta_phi_sq[i] = pow(coordif[i][4] - phi0 - Ex * coordif[i][0] - Ey * coordif[i][1] - Ez * coordif[i][2], 2);
        delta_phi_sq_E += delta_phi_sq[i] / n_group / 4;
    }
    fullnes = 0;
    prew_fullnes = n_group * 4;
    while (fullnes != prew_fullnes)
    {
        prew_fullnes = fullnes;
        fullnes = 0;
        delta_phi_sq_E_iter = 0;
        for (i = 0; i < n_group * 4; i++)
            if (delta_phi_sq[i] < 9 * delta_phi_sq_E)
            {
                delta_phi_sq_E_iter += delta_phi_sq[i];
                fullnes++;
            }
        delta_phi_sq_E = delta_phi_sq_E_iter / fullnes;
    }
    return delta_phi_sq_E;
}

static void LSMcontroller(double** coordif, double FEEx[3], double FEDis[3], double ef[3])
{
    int i, j, k, m;
    int opt_i = 0, opt_j = 0, opt_k = 0, opt_m = 0;
    double phi_max = -60;
    double phi_min = 60;
    double Ex, Ey, Ez, phi0;
    double delta_phi;
    double delta_phi_opt = 1e6;
    for (i = 0; i < n_group * 4; i++)
    {
        if (coordif[i][4] < phi_min)
            phi_min = coordif[i][4];
        if (coordif[i][4] > phi_max)
            phi_max = coordif[i][4];
    }
    for (i = 0; i < LSM_iter_param_x; i++)
        for (j = 0; j < LSM_iter_param_y; j++)
            for (k = 0; k < LSM_iter_param_z; k++)
                for (m = 0; m < LSM_iter_phi; m++)
                {
                    //printf(" % i\t % i\n", i, m);
                    Ex = FEEx[0] - FEDis[0] * (2 - 4 * i / (float)LSM_iter_param_x);
                    Ey = FEEx[1] - FEDis[1] * (2 - 4 * j / (float)LSM_iter_param_y);
                    Ez = FEEx[2] - FEDis[2] * (2 - 4 * k / (float)LSM_iter_param_z);
                    phi0 = phi_min + (phi_max - phi_min) * m / (float)LSM_iter_phi;
                    delta_phi = LSMcalcus(coordif, Ex, Ey, Ez, phi0);
                    if (delta_phi < delta_phi_opt)
                    {
                        delta_phi_opt = delta_phi;
                        opt_i = i;
                        opt_j = j;
                        opt_k = k;
                        opt_m = m;
                    }
                }
    ef[0] = Ex = FEEx[0] - FEDis[0] * (2 - 4 * opt_i / (float)LSM_iter_param_x);
    ef[1] = Ey = FEEx[1] - FEDis[1] * (2 - 4 * opt_j / (float)LSM_iter_param_y);
    ef[2] = Ez = FEEx[2] - FEDis[2] * (2 - 4 * opt_k / (float)LSM_iter_param_z);

}

int inv_matrix(double* A, size_t N)
{
    int i, j, k;
    double max_val;
    double factor;
    int max_row;
    double** B;
    B = (double**)malloc(N * sizeof(double*));
    for (i = 0; i < N; i++)
        B[i] = (double*)malloc(N * sizeof(double));
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            if (i == j)
                B[i][j] = 1;
            else
                B[i][j] = 0;
    // Приведем матрицу к виду, наболее близкому к диагональному, перестановкой строк
    for (i = 0; i < N - 1; i++)
    {
        max_row = i;
        max_val = fabs(A[i * N + i]);
        for (j = i + 1; j < N; j++) {
            if (fabs(A[j * N + i]) > max_val) {
                max_val = fabs(A[j * N + i]);
                max_row = j;
            }
        }
        if (i != max_row)
        {
            for (j = 0; j < N; j++)
            {
                A[i * N + j] = A[max_row * N + j] - A[i * N + j];
                A[max_row * N + j] = A[max_row * N + j] - A[i * N + j];
                A[i * N + j] = A[max_row * N + j] + A[i * N + j];
                B[i][j] = B[max_row][j] - B[i][j];
                B[max_row][j] = B[max_row][j] - B[i][j];
                B[i][j] = B[max_row][j] + B[i][j];
            }
        }
        // Учет вырожденности
        if (max_val < 1e-12)
        {
            printf("Error max_val almost equal to 0, try using old method for this cases");
            return -1;

        }
        //printf("%lf\t%lf\t%lf\n", A[0][0], A[0][1], A[0][2]);
        //printf("%lf\t%lf\t%lf\n", A[1][0], A[1][1], A[1][2]);
        //printf("%lf\t%lf\t%lf\n\n", A[2][0], A[2][1], A[2][2]);
        for (j = i + 1; j < N; j++) {
            factor = A[j * N + i] / A[i * N + i];
            for (k = i; k < N; k++) {
                A[j * N + k] -= factor * A[i * N + k];
                B[j][k] -= factor * B[i][k];
            }
        }

    }
    //printf("%lf\t%lf\t%lf\n", delta_phi[0], delta_phi[1], delta_phi[2]);
    //printf("%lf\t%lf\t%lf\n", A[0][0], A[0][1], A[0][2]);
    //printf("%lf\t%lf\t%lf\n", A[1][0], A[1][1], A[1][2]);
    //printf("%lf\t%lf\t%lf\n\n", A[2][0], A[2][1], A[2][2]);
    for (i = N - 1; i >= 0; i--)
    {
        for (j = i + 1; j < N; j++)
        {
            for (k = 0; k < N; k++)
                B[i][k] -= B[j][k] * A[i * N + j];
            A[i * N + j] = 0;
        }
        for (j = 0; j < N; j++)
        {
            B[i][j] /= A[i * N + i];
        }
        A[i * N + i] = 1;
    }
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            A[i * N + j] = B[i][j];
    for (i = 0; i < N; i++)
        free(B[i]);
    free(B);
    return 0;
}

void gaus3x3solver(double A[3][3], double ef[3], double delta_phi[3])
{
    int i, j, k;
    double E[3];
    // Решение будем производить методом Гаусса
    double max_val;
    double factor;
    int max_row;
    // Приведем матрицу к виду, наболее близкому к диагональному, перестановкой строк
    for (i = 0; i < 2; i++)
    {
        max_row = i;
        max_val = fabs(A[i][i]);

        for (j = i + 1; j < 3; j++) {
            if (fabs(A[j][i]) > max_val) {
                max_val = fabs(A[j][i]);
                max_row = j;
            }
        }
        if (i != max_row)
        {
            A[i][0] = A[max_row][0] - A[i][0];
            A[max_row][0] = A[max_row][0] - A[i][0];
            A[i][0] = A[max_row][0] + A[i][0];
            A[i][1] = A[max_row][1] - A[i][1];
            A[max_row][1] = A[max_row][1] - A[i][1];
            A[i][1] = A[max_row][1] + A[i][1];
            A[i][2] = A[max_row][2] - A[i][2];
            A[max_row][2] = A[max_row][2] - A[i][2];
            A[i][2] = A[max_row][2] + A[i][2];
            delta_phi[i] = delta_phi[max_row] - delta_phi[i];
            delta_phi[max_row] = delta_phi[max_row] - delta_phi[i];
            delta_phi[i] = delta_phi[max_row] + delta_phi[i];
        }
        // Учет вырожденности
        if (max_val < 1e-2)
        {
            printf("Error max_val almost equal to 0");
            ef[0] = 0;
            ef[1] = 0;
            ef[2] = 0;
            return;

        }
        //printf("%lf\t%lf\t%lf\n", A[0][0], A[0][1], A[0][2]);
        //printf("%lf\t%lf\t%lf\n", A[1][0], A[1][1], A[1][2]);
        //printf("%lf\t%lf\t%lf\n\n", A[2][0], A[2][1], A[2][2]);
        for (j = i + 1; j < 3; j++) {
            factor = A[j][i] / A[i][i];
            for (k = i; k < 3; k++) {
                A[j][k] -= factor * A[i][k];
            }
            delta_phi[j] -= factor * delta_phi[i];
        }

    }
    //printf("%lf\t%lf\t%lf\n", delta_phi[0], delta_phi[1], delta_phi[2]);
    //printf("%lf\t%lf\t%lf\n", A[0][0], A[0][1], A[0][2]);
    //printf("%lf\t%lf\t%lf\n", A[1][0], A[1][1], A[1][2]);
    //printf("%lf\t%lf\t%lf\n\n", A[2][0], A[2][1], A[2][2]);
    for (i = 2; i >= 0; i--)
    {
        E[i] = delta_phi[i];
        for (j = i + 1; j < 3; j++) {
            E[i] -= A[i][j] * E[j];
        }
        E[i] /= A[i][i];
    }
    ef[0] = E[0];
    ef[1] = E[1];
    ef[2] = E[2];


}

static void get_ef(Mytree* a, double n, double y1[6], double ef[3])
{
    size_t num_neighbors = n_group*4;
    int i, j, k;
    std::vector<size_t> indices(num_neighbors);
    std::vector<double> distances(num_neighbors);
    double x = L * y1[0];
    double z = L * y1[4];
    double y = L * y1[2];
    double query_point[3] = { x, y, z };
    size_t found = (*a).knnSearch(
        query_point,           // точка запроса
        num_neighbors,         // сколько соседей
        &indices[0],           // массив для индексов
        &distances[0]          // массив для расстояний
    );
    //Нахождение ближайшего узла и ещё 3
    double** coordif;
    coordif = (double**)malloc(4*n_group * sizeof(double*));
    for (i = 0; i < 4*n_group; i++)
        coordif[i] = (double*)malloc(4 * sizeof(double));
    for (i = 0; i < 4*n_group; i++)
    {
        coordif[i][0] = -x + (*a).dataset_.kdtree_get_pt(indices[i], 0);
        coordif[i][1] = -y + (*a).dataset_.kdtree_get_pt(indices[i], 1);
        coordif[i][2] = -z + (*a).dataset_.kdtree_get_pt(indices[i], 2);
        coordif[i][3] = (*a).dataset_.kdtree_get_pt(indices[i], 3);
    }
    double A[n_group * 4 * 4];
    double A_t[4 * n_group * 4];
    double A_res[4 * 4 * n_group];
    double A_inv[16];
    double solution[4];
    double y_mat[n_group * 4];
    ef[0] = 0;
    ef[1] = 0;
    ef[2] = 0;
    for (i = 0; i < n_group * 4; i++)
    {
        for (j = 0; j < 3; j++)
            A[i * 4 + j] = coordif[i][j];
        A[i * 4 + 3] = 1;
        y_mat[i] = coordif[i][3];
    }
    Transposition(A_t, A, n_group * 4, 4);
    multiplier(A_inv, A_t, A, 4, n_group * 4, 4);
    k = inv_matrix(A_inv, 4);
    if (k == -1)
        return;
    multiplier(A_res, A_inv, A_t, 4, 4, n_group * 4);
    multiplier(solution, A_res, y_mat, 4, n_group * 4, 1);
    ef[0] = solution[0];
    ef[1] = solution[1];
    ef[2] = solution[2];
    for (i = 0; i < n_group*4; i++)
        free(coordif[i]);
    free(coordif);
}

static void get_ef_old(double** a, size_t n, double y1[6], double ef[3])
{
    int i, j, k;
    int fullnes;
    int prew_fullnes;
    double x = L * y1[0];
    double z = L * y1[4];
    double y = L * y1[2];
    double exE[3];
    double DisE[3];
    double DisE2[3];
    //Нахождение ближайшего узла и ещё 3
    double** coordif;
    coordif = (double**)malloc(n * sizeof(double*));
    for (i = 0; i < n; i++)
        coordif[i] = (double*)malloc(5 * sizeof(double));
    for (i = 0; i < n; i++)
    {
        coordif[i][0] = -x + a[i][0];
        coordif[i][1] = -y + a[i][1];
        coordif[i][2] = -z + a[i][2];
        coordif[i][3] = pow(pow(coordif[i][0], 2) + pow(coordif[i][1], 2) + pow(coordif[i][2], 2), 0.5);
        coordif[i][4] = a[i][3];
    }
    for (j = 0; j < n_group * 4; j++)
    {
        for (i = j; i < n; i++)
        {
            if (coordif[j][3] > coordif[i][3])
            {
                coordif[j][4] = coordif[i][4] - coordif[j][4];
                coordif[i][4] = coordif[i][4] - coordif[j][4];
                coordif[j][4] = coordif[j][4] + coordif[i][4];
                coordif[j][3] = coordif[i][3] - coordif[j][3];
                coordif[i][3] = coordif[i][3] - coordif[j][3];
                coordif[j][3] = coordif[j][3] + coordif[i][3];
                coordif[j][2] = coordif[i][2] - coordif[j][2];
                coordif[i][2] = coordif[i][2] - coordif[j][2];
                coordif[j][2] = coordif[j][2] + coordif[i][2];
                coordif[j][1] = coordif[i][1] - coordif[j][1];
                coordif[i][1] = coordif[i][1] - coordif[j][1];
                coordif[j][1] = coordif[j][1] + coordif[i][1];
                coordif[j][0] = coordif[i][0] - coordif[j][0];
                coordif[i][0] = coordif[i][0] - coordif[j][0];
                coordif[j][0] = coordif[j][0] + coordif[i][0];

            }
        }
    }
    //Вычисление разности потенциалов между ближайшим и ещё 3
    double delta_phi[3];
    double A[3][3];
    double E[n_group][3];
    ef[0] = 0;
    ef[1] = 0;
    ef[2] = 0;
    for (i = 0; i < n_group; i++)
    {
        delta_phi[0] = coordif[1 * n_group + i][4] - coordif[0 * n_group + i][4];
        delta_phi[1] = coordif[2 * n_group + i][4] - coordif[0 * n_group + i][4];
        delta_phi[2] = coordif[3 * n_group + i][4] - coordif[0 * n_group + i][4];
        //Составление системы Ax=b,
        // где b - разность потенциаловб, 
        // x - искомый градиент (E),
        // A матрица каждая строка которой соотвествует паре точек (как и b) и имеет вид:
        // delta x, delta y, delta z где все эти дельты - разности координат точек, для удобства будем всегда брать конечную минус начальную
        A[0][0] = coordif[1 * n_group + i][0] - coordif[0 * n_group + i][0];
        A[1][0] = coordif[2 * n_group + i][0] - coordif[0 * n_group + i][0];
        A[2][0] = coordif[3 * n_group + i][0] - coordif[0 * n_group + i][0];
        A[0][1] = coordif[1 * n_group + i][1] - coordif[0 * n_group + i][1];
        A[1][1] = coordif[2 * n_group + i][1] - coordif[0 * n_group + i][1];
        A[2][1] = coordif[3 * n_group + i][1] - coordif[0 * n_group + i][1];
        A[0][2] = coordif[1 * n_group + i][2] - coordif[0 * n_group + i][2];
        A[1][2] = coordif[2 * n_group + i][2] - coordif[0 * n_group + i][2];
        A[2][2] = coordif[3 * n_group + i][2] - coordif[0 * n_group + i][2];
        //printf("%lf\t%lf\t%lf\n", delta_phi[0], delta_phi[1], delta_phi[2]);
        //printf("%lf\t%lf\t%lf\n", A[0][0], A[0][1], A[0][2]);
        //printf("%lf\t%lf\t%lf\n", A[1][0], A[1][1], A[1][2]);
        //printf("%lf\t%lf\t%lf\n\n", A[2][0], A[2][1], A[2][2]);
        gaus3x3solver(A, E[i], delta_phi);
    }
    /*
    for (i = 0; i < n_group; i++)
    {
        ef[0] += E[i][0] / (float)n_group;
        ef[1] += E[i][1] / (float)n_group;
        ef[2] += E[i][2] / (float)n_group;
    }
    */
    DisE[0] = 0;
    DisE[1] = 0;
    DisE[2] = 0;
    exE[0] = 0;
    exE[1] = 0;
    exE[2] = 0;
    for (i = 0; i < n_group; i++)
    {
        exE[0] += E[i][0] / (float)n_group;
        exE[1] += E[i][1] / (float)n_group;
        exE[2] += E[i][2] / (float)n_group;
    }
    for (i = 0; i < n_group; i++)
    {
        DisE[0] += pow((E[i][0] - exE[0]), 2) / (float)n_group;
        DisE[1] += pow((E[i][1] - exE[1]), 2) / (float)n_group;
        DisE[2] += pow((E[i][2] - exE[2]), 2) / (float)n_group;
    }

    DisE[0] = pow(DisE[0], 0.5);
    DisE[1] = pow(DisE[1], 0.5);
    DisE[2] = pow(DisE[2], 0.5);
    for (i = 0; i < n_group; i++)
    {
        ef[0] += 0;
        ef[1] += 0;
        ef[2] += 0;
    }
    fullnes = 0;
    prew_fullnes = 1;
    DisE2[0] = 0;
    DisE2[1] = 0;
    DisE2[2] = 0;
    while (fullnes != prew_fullnes)
    {
        prew_fullnes = fullnes;
        fullnes = 0;
        for (i = 0; i < n_group; i++)
            if ((fabs(exE[0] - E[i][0]) <= 2 * DisE[0]) && (fabs(exE[1] - E[i][1]) <= 2 * DisE[1]) && (fabs(exE[2] - E[i][2]) <= 2 * DisE[2]))
            {
                ef[0] += E[i][0];
                ef[1] += E[i][1];
                ef[2] += E[i][2];
                DisE2[0] += pow((E[i][0] - exE[0]), 2);
                DisE2[1] += pow((E[i][1] - exE[1]), 2);
                DisE2[2] += pow((E[i][2] - exE[2]), 2);
                fullnes++;
            }
        DisE[0] = pow(DisE2[0] / fullnes, 0.5);
        DisE[1] = pow(DisE2[1] / fullnes, 0.5);
        DisE[2] = pow(DisE2[2] / fullnes, 0.5);
        DisE2[0] = 0;
        DisE2[1] = 0;
        DisE2[2] = 0;
        exE[0] = ef[0] / fullnes;
        exE[1] = ef[1] / fullnes;
        exE[2] = ef[2] / fullnes;
        ef[0] = 0;
        ef[1] = 0;
        ef[2] = 0;
    }
    LSMcontroller(coordif, exE, DisE, ef);
    /*
    exE[0] = 0;
    exE[1] = 0;
    exE[2] = 0;
    DisE[0] = 0;
    DisE[1] = 0;
    DisE[2] = 0;
    DisE2[0] = 0;
    DisE2[1] = 0;
    DisE2[2] = 0;
    for (i = 0; i < n_group; i++)
    {
        exE[0] += E[i][0]/(float)n_group;
        exE[1] += E[i][1]/(float)n_group;
        exE[2] += E[i][2]/(float)n_group;
    }
    for (i = 0; i < n_group; i++)
    {
        DisE[0] += pow((E[i][0]-exE[0]),2)/(float)n_group;
        DisE[1] += pow((E[i][1]-exE[1]),2)/(float)n_group;
        DisE[2] += pow((E[i][2]-exE[2]),2)/(float)n_group;
    }
    DisE[0] = pow(DisE[0], 0.5);
    DisE[1] = pow(DisE[1], 0.5);
    DisE[2] = pow(DisE[2], 0.5);
    for (i = 0; i < n_group; i++)
    {
        if (((E[i][0] - exE[0]) > 1.5*DisE[0])&&(E[i][0]>0))
        {
            A[0][0] = coordif[1 * n_group + i][0] - coordif[0 * n_group + i][0];
            A[1][0] = coordif[2 * n_group + i][0] - coordif[0 * n_group + i][0];
            A[2][0] = coordif[3 * n_group + i][0] - coordif[0 * n_group + i][0];
            A[0][1] = coordif[1 * n_group + i][1] - coordif[0 * n_group + i][1];
            A[1][1] = coordif[2 * n_group + i][1] - coordif[0 * n_group + i][1];
            A[2][1] = coordif[3 * n_group + i][1] - coordif[0 * n_group + i][1];
            A[0][2] = coordif[1 * n_group + i][2] - coordif[0 * n_group + i][2];
            A[1][2] = coordif[2 * n_group + i][2] - coordif[0 * n_group + i][2];
            A[2][2] = coordif[3 * n_group + i][2] - coordif[0 * n_group + i][2];
            delta_phi[0] = coordif[1 * n_group + i][4] - coordif[0 * n_group + i][4];
            delta_phi[1] = coordif[2 * n_group + i][4] - coordif[0 * n_group + i][4];
            delta_phi[2] = coordif[3 * n_group + i][4] - coordif[0 * n_group + i][4];
            gaus3x3solver(A, E[i], delta_phi);
            printf("%lf\t%lf\t%lf\n", E[i][0], E[i][1], E[i][2]);
            printf("%lf\t%lf\t%lf\n", delta_phi[0], delta_phi[1], delta_phi[2]);
            printf("%lf\t%lf\t%lf\n", A[0][0], A[0][1], A[0][2]);
            printf("%lf\t%lf\t%lf\n", A[1][0], A[1][1], A[1][2]);
            printf("%lf\t%lf\t%lf\n\n", A[2][0], A[2][1], A[2][2]);
        }
    }
    j = 0;
    for (i = 0; i < n_group; i++)
    {
        if ((fabs(E[i][0] - exE[0]) < DisE[0]))
        {
            ef[0] += E[i][0];
            ef[1] += E[i][1];
            ef[2] += E[i][2];
            j++;
            DisE2[0] += pow((E[i][0] - exE[0]), 2);
            DisE2[1] += pow((E[i][1] - exE[1]), 2);
            DisE2[2] += pow((E[i][2] - exE[2]), 2);
        }
    }
    exE[0] = ef[0] / j;
    exE[1] = ef[1] / j;
    exE[2] = ef[2] / j;
    ef[0] = 0;
    ef[1] = 0;
    ef[2] = 0;
    DisE2[0] = pow(DisE2[0]/j, 0.5);
    DisE2[1] = pow(DisE2[1]/j, 0.5);
    DisE2[2] = pow(DisE2[2]/j, 0.5);
    j = 0;
    for (i = 0; i < n_group; i++)
    {
        if ((fabs(E[i][0] - exE[0]) < DisE2[0]))
        {
            ef[0] += E[i][0];
            ef[1] += E[i][1];
            ef[2] += E[i][2];
            j++;
            DisE[0] += pow((E[i][0] - exE[0]), 2);
            DisE[1] += pow((E[i][1] - exE[1]), 2);
            DisE[2] += pow((E[i][2] - exE[2]), 2);
        }
    }
    ef[0] = ef[0] / j;
    ef[1] = ef[1] / j;
    ef[2] = ef[2] / j;
    */
    for (i = 0; i < n; i++)
        free(coordif[i]);
    free(coordif);
}

/*
void get_fi(double** a, double n, double y1[6], double *phi)
{
    int i, j, k;
    double summ;
    double x = L * y1[0];
    double z = L * y1[4];
    double y = L * y1[2];
    //Нахождение ближайшего узла и ещё 3
    double** coordif;
    coordif = (double**)malloc(n * sizeof(double*));
    for (i = 0; i < n; i++)
        coordif[i] = (double*)malloc(5 * sizeof(double));
    for (i = 0; i < n; i++)
    {
        coordif[i][0] = -x + a[i][0];
        coordif[i][1] = -y + a[i][1];
        coordif[i][2] = -z + a[i][2];
        coordif[i][3] = pow(pow(coordif[i][0], 2) + pow(coordif[i][1], 2) + pow(coordif[i][2], 2), 0.5);
        coordif[i][4] = a[i][3];
    }
    for (j = 0; j < n_inter; j++)
    {
        for (i = j; i < n; i++)
        {
            if (coordif[j][3] > coordif[i][3])
            {
                coordif[j][4] = coordif[i][4] - coordif[j][4];
                coordif[i][4] = coordif[i][4] - coordif[j][4];
                coordif[j][4] = coordif[j][4] + coordif[i][4];
                coordif[j][3] = coordif[i][3] - coordif[j][3];
                coordif[i][3] = coordif[i][3] - coordif[j][3];
                coordif[j][3] = coordif[j][3] + coordif[i][3];
                coordif[j][2] = coordif[i][2] - coordif[j][2];
                coordif[i][2] = coordif[i][2] - coordif[j][2];
                coordif[j][2] = coordif[j][2] + coordif[i][2];
                coordif[j][1] = coordif[i][1] - coordif[j][1];
                coordif[i][1] = coordif[i][1] - coordif[j][1];
                coordif[j][1] = coordif[j][1] + coordif[i][1];
                coordif[j][0] = coordif[i][0] - coordif[j][0];
                coordif[i][0] = coordif[i][0] - coordif[j][0];
                coordif[j][0] = coordif[j][0] + coordif[i][0];

            }
        }
    }
    printf("%lf\t%lf\t%lf\t%lf\n", coordif[0][3], coordif[1][3], coordif[2][3], coordif[3][3]);
    printf("%lf\t%lf\t%lf\t%lf\n", coordif[0][4], coordif[1][4], coordif[2][4], coordif[3][4]);
    //Вычисление потенциала в точке
    *phi = 0;
    summ = 0;
    for (i = 0; i < n_inter; i++)
    {
        *phi += coordif[i][4] / coordif[i][3];
        summ += 1 / coordif[i][3];
    }
    printf("%lf\n", phi);
    *phi = *phi / summ;
}
*/
int main()
{
    double** EF;
    int i, j, k;
    int n_epoints_temp;
    size_t n_epoints;
    FILE* fel;
    fel = fopen("Efield", "r");
    if (fel == NULL)
    {
        printf("\nError - reading of EF is impossible!");
        return -1;
    };
    fscanf(fel, "%i", &n_epoints_temp);
    n_epoints = (size_t)n_epoints_temp;
    EF = (double**)malloc(n_epoints * sizeof(double*));
    for (j = 0; j < n_epoints; j++)
        EF[j] = (double*)malloc(4 * sizeof(double));
    j = 0;
    i = 1;
    while (i == 1)
    {
        i = fscanf(fel, "%lf", &EF[j / 4][j % 4]);
        j++;
    }
    printf("%zu", n_epoints);
    /*
    for (i = 0; i < n_epoints; i++)
    {
        printf("%lf\t%lf\t%lf\t%lf\n", EF[i][0], EF[i][1], EF[i][2], EF[i][3]);
    }
    */
    Adapter adapter(EF, n_epoints);
    Mytree* kdtree;
    kdtree = new Mytree(3, adapter);
    (*kdtree).buildIndex();
    // Создаешь дерево
    FILE* fw;
    char fwname[30];
    /*int n0;
    printf("\nn0=...");
    scanf("%d", &n0);*/
    double r[6] = { 0, 0, 0, 0, -2.5, 0 };
    double Ans[1201][3];
    fw = fopen("check", "w");
#pragma omp parallel for
    for (i = 0; i < 1201; i++)
    {
        double local_ef[3], local_r[6];
        memcpy(local_r, r, 6 * sizeof(double));
        local_r[4] += i * (5.0 / L);
        get_ef( kdtree, n_epoints, local_r, local_ef);
        //ordered can be changed to critical for faster calculations if order is not required
        Ans[i][0] = local_ef[0];
        Ans[i][1] = local_ef[1];
        Ans[i][2] = local_ef[2];
        //               fprintf(fw, "%i\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", i, local_mf[0], local_mf[1], local_mf[2], local_r[0], local_r[2], local_r[4]);
        printf("%i\t", i);
    }
    for (i = 0; i < 1201; i++)
    {
        fprintf(fw, "%i\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", i, Ans[i][0], Ans[i][1], Ans[i][2], 0 , 0, -2.5 + i * (5.0 / L));
    }
/*
    for (k = 0; k < 57; k++)
    {
        sprintf(fwname, "Ef_%d.dat", (k*5));
        fw = fopen(fwname, "w");
        for (j = 0; j < 113; j++)
        {
#pragma omp parallel for
            for (i = 0; i < 1201; i++)
            {
                double local_ef[3], local_r[6];
                memcpy(local_r, r, 6 * sizeof(double));
                local_r[4] += i * (5.0 / L);
                get_ef(EF, n_epoints, local_r, local_ef);
                //ordered can be changed to critical for faster calculations if order is not required
                Ans[i][0] = local_ef[0];
                Ans[i][1] = local_ef[1];
                Ans[i][2] = local_ef[2];
                //               fprintf(fw, "%i\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", i, local_mf[0], local_mf[1], local_mf[2], local_r[0], local_r[2], local_r[4]);
                               printf("%i\t", i);
            }
            for (i = 0; i < 1201; i++)
            {
                fprintf(fw, "%lf\t%lf\t%lf\n", Ans[i][0], Ans[i][1], Ans[i][2]);
            }
            printf("\n\n%i\t%i\n\n", k, j);
            r[0] += 5.0/L;
        }
        r[2] += 5.0 / L;
        fclose(fw);
    }
*/
}