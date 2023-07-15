#pragma once
#include <vector>
#include "type.hpp"

using namespace std;

class tdarray {
public:
    tdarray() = default;

    tdarray(vector<Real64> &data);

    tdarray(vector<vector<Real64>> &data);

    tdarray(UInt row, UInt col);

    ~tdarray() = default;

    /**
     * 将二维数组随机为均值为 0，标准差为 1 的正态分布
     */
    void random();

    /**
     * 与二维数组b作点积运算，结果保存在result中
     */
    void dot(const tdarray &b, tdarray &result) const;

    /**
     * 根据指定维度将数组转换为经过softmax函数处理的结果
     * axis == 0为第一维度，axis == 1为第二维度
     */
    void softmax(UInt axis);

    /**
     * 返回二维数组的形状
     */
    pair<UInt, UInt> shape() const {
        return {row_, col_};
    };

    vector<vector<Real64>>::iterator begin() {
        return data_.begin();
    }

    vector<vector<Real64>>::iterator end() {
        return data_.end();
    }

    Real64 &operator()(UInt row, UInt col) {
        return data_[row][col];
    }

    const Real64 &operator()(UInt row, UInt col) const {
        return data_[row][col];
    }

    /**
     * 与维度相同的二维数组对应元素相加的结果
     */
    tdarray operator+(const tdarray &b) const;

    /**
     * 与维度相同的二维数组对应元素相减的结果
     */
    tdarray operator-(const tdarray &b) const;

    /**
     * 逐行打印二维数组
     */
    void log() const;
private:
    vector<vector<Real64>> data_;
    UInt row_;
    UInt col_;
};
