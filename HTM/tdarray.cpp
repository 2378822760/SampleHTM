//
// Created by 曹 on 2023/7/11.
//
#include <random>
#include <iostream>
#include <cmath>
#include "tdarray.hpp"

tdarray::tdarray(vector<Real64> &data) {
    data_.emplace_back(data.begin(), data.end());
    row_ = 1, col_ = data.size();
}

tdarray::tdarray(vector<vector<Real64>> &data) {
    data_ = data;
    row_ = data.size(), col_ = data[0].size();
}

tdarray::tdarray(UInt row, UInt col) {
    data_ = vector<vector<Real64>>(row, vector<Real64>(col, 0));
    row_ = row, col_ = col;
}

void tdarray::random() {
    random_device rd;  // 用于获取随机种子
    mt19937 gen(rd()); // 使用 Mersenne Twister 算法的随机数引擎
    // 均值为 0，标准差为 1 的正态分布
    normal_distribution<Real64> dist(0.0, 1.0);
    // 随机初始化权重
    for (auto &row : data_) {
        for (auto &col : row) {
            col = dist(gen);
        }
    }
}

void tdarray::dot(const tdarray &b, tdarray &result) const {
    if (col_ != b.row_) {
        throw runtime_error("the shape of two tdarray is not match!");
        return;
    }
    result = tdarray(row_, b.col_);
    for (UInt i = 0; i < row_; ++i) {
        for (UInt j = 0; j < b.col_; ++j) {
            for (UInt k = 0; k < col_; ++k) {
                result(i, j) += data_[i][k] * b(k, j);
            }
        }
    }
}

void tdarray::log() const {
    for (auto &row : data_) {
        for (auto &col : row) {
            cout << col << " ";
        }
        cout << endl;
    }
}

void tdarray::softmax(UInt axis) {
    for (auto& row : data_) {
        for (auto& n : row) {
            n = exp(n);
        }
    }
    if (axis == 0) {
        for (UInt i = 0; i < row_; ++i) {
            Real64 sum = 0;
            for (UInt j = 0; j < col_; ++j) {
                sum += data_[i][j];
            }
            for (UInt j = 0; j < col_; ++j) {
                data_[i][j] /= sum;
            }
        }
    } else if (axis == 1) {
        for (UInt i = 0; i < col_; ++i) {
            Real64 sum = 0;
            for (UInt j = 0; j < row_; ++j) {
                sum += data_[j][i];
            }
            for (UInt j = 0; j < row_; ++j) {
                data_[j][i] /= sum;
            }
        }
    }
}

tdarray tdarray::operator+(const tdarray &b) const {
    if (row_ != b.row_ || col_ != b.col_) {
        throw runtime_error("the shape of two tdarray is not match!");
    }
    tdarray result(row_, col_);
    for (UInt i = 0; i < row_; ++i) {
        for (UInt j = 0; j < col_; ++j) {
            result(i, j) = data_[i][j] + b(i, j);
        }
    }
    return result;
}

tdarray tdarray::operator-(const tdarray &b) const {
    if (row_ != b.row_ || col_ != b.col_) {
        throw runtime_error("the shape of two tdarray is not match!");
    }
    tdarray result(row_, col_);
    for (UInt i = 0; i < row_; ++i) {
        for (UInt j = 0; j < col_; ++j) {
            result(i, j) = data_[i][j] - b(i, j);
        }
    }
    return result;
}

