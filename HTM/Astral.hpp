#pragma once

#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace Astral {
    class ActivationBase {
    public:
        virtual Eigen::MatrixXd f(const Eigen::MatrixXd& units, double a = 1) = 0;
        virtual Eigen::MatrixXd df(const Eigen::MatrixXd& units, double a = 1) = 0;
        virtual Eigen::MatrixXd dLdz(const Eigen::MatrixXd& z, const Eigen::MatrixXd& dLdy) = 0;
    };

    class LossBase {
    public:
        virtual double f(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target) = 0;
        virtual Eigen::MatrixXd df(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target) = 0;
    };

    class OptimizerBase {
    public:
        virtual void update(Eigen::MatrixXd& params, const Eigen::MatrixXd& grads) = 0;
    };

    class Sigmoid : public ActivationBase {
    public:
        Eigen::MatrixXd f(const Eigen::MatrixXd& units, double a = 1) override {
            return units.unaryExpr([](double elem) {
                return 1 / (1 + std::exp(-elem));
                });
        }

        Eigen::MatrixXd df(const Eigen::MatrixXd& units, double a = 1) override {
            return units.unaryExpr([](double elem) {
                return std::exp(-elem) / std::pow(1.0 + std::exp(-elem), 2);
                });
        }

        Eigen::MatrixXd dLdz(const Eigen::MatrixXd& z, const Eigen::MatrixXd& dLdy) {
            Eigen::MatrixXd dydz = df(z);
            return dydz.array() * dLdy.array();
        }
    };

    class Softmax : public ActivationBase {
    public:
        Eigen::MatrixXd f(const Eigen::MatrixXd& units, double a = 1) override {
            auto res = units.array().exp();
            Eigen::VectorXd sumExp = res.rowwise().sum();
            return res.array().colwise() / sumExp.array();
        }

        Eigen::MatrixXd df(const Eigen::MatrixXd& units, double a = 1) override {
            Eigen::MatrixXd Jacobi = -units.transpose() * units;
            for (int i = 0; i < units.cols(); ++i) Jacobi(i, i) += units(0, i); // Ö÷¶Ô½ÇÏß + pi
            return Jacobi;
        }

        Eigen::MatrixXd dLdz(const Eigen::MatrixXd& z, const Eigen::MatrixXd& dLdy) {
            Eigen::MatrixXd y = f(z);
            Eigen::MatrixXd res(y.rows(), y.cols());
            // std::cout << dLdy << std::endl;
            auto batchsize = y.rows();
            for (Eigen::Index i = 0; i < batchsize; ++i) {
                res.row(i) = dLdy.row(i) * df(y.row(i));
            }
            return res;
        }
    };


    class ReLU : public ActivationBase {
    public:
        Eigen::MatrixXd f(const Eigen::MatrixXd& units, double a = 1) override {
            return units.array().max(0.0);
        }
        Eigen::MatrixXd df(const Eigen::MatrixXd& units, double a = 1) override {
            return (units.array() >= 0.0).cast<double>();
        }
        Eigen::MatrixXd dLdz(const Eigen::MatrixXd& z, const Eigen::MatrixXd& dLdy) override {
            Eigen::MatrixXd y = f(z);
            Eigen::MatrixXd dydz = df(y);
            return dydz.array() * dLdy.array();
        }
    };


    class MSELoss : public LossBase {
    public:
        double f(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target) {
            auto res = (target - input).array().square();
            return res.sum() / res.size();
        }

        Eigen::MatrixXd df(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target) {
            return (target - input) * 2.0 / target.size();
        }
    };


    class CrossEntropyLoss : public LossBase {
    public:
        double f(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target) {
            auto res = input.array().log() * target.array();
            return -res.sum() / input.rows();
        }

        Eigen::MatrixXd df(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target) {
            return -target.array() * input.array().inverse() / input.rows();
        }
    };


    class SGD : public OptimizerBase {
    public:
        double lr_;

        SGD(double lr) : lr_(lr) {};

        void setLr(double lr) { this->lr_ = lr; };

        void update(Eigen::MatrixXd& params, const Eigen::MatrixXd& grads) override {
            params -= lr_ * grads;
        }
    };


    enum LayerType {
        Input,
        Hidden,
        Output
    };

    class Layer {
    public:
        Layer(int unitsNum, ActivationBase* activation, LayerType type);

        Layer(const Layer&) = default;

        Layer(Layer&&) = default;

        Layer& operator=(const Layer&) = default;

        Layer& operator=(Layer&&) = default;

        Eigen::MatrixXd y_;

        Eigen::MatrixXd z_;

        Eigen::MatrixXd w_;

        Eigen::MatrixXd b_;

        void log();

        int unitsNum_;

        ActivationBase* activation_;

        LayerType type_;
    }; // Layer

    class SelfAttention {
        SelfAttention(int embedSize, int heads);

        void compile();

        Eigen::MatrixXd forward(const Eigen::MatrixXd inputs);

        void backward();

        int embedSize_, heads;

        Eigen::MatrixXd Wq_, Wk_, Wv_;

        Eigen::MatrixXd Q_, K_, V_, Alpha_, Alpha2_;
    };

    class Sequential {
    public:
        Sequential() = default;

        Sequential(const Sequential&) = default;

        Sequential(Sequential&&) = default;

        Sequential& operator=(const Sequential&) = default;

        Sequential& operator=(Sequential&&) = default;

        void add(int unitsNum, ActivationBase* activation, LayerType type = LayerType::Hidden);

        void pop();

        void compile(OptimizerBase* optimizer, LossBase* lossfunc);

        void fit(const Eigen::MatrixXd& x_train, const Eigen::MatrixXd& y_train, int batchSize, int epochs);

        double evaluate(const Eigen::MatrixXd& x_train, const Eigen::MatrixXd& y_train);

        Eigen::MatrixXd predict(const Eigen::MatrixXd& x);

        void forwardPropagation(const Eigen::MatrixXd& input);

        double calculateLoss(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target, LossBase* lossfunc);

        void backPropagation(const Eigen::MatrixXd& target);

        void log();

        std::vector<Layer> layers_;

        OptimizerBase* optimizer_;

        LossBase* lossfunc_;
    }; // Seqiential
    
} // Astral