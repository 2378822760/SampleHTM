#pragma once

#include <iostream>
#include <vector>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>
// #define DOUBLE_TYPE
// #define FLOAT_TYPE

#ifdef FLOAT_TYPE
typedef Eigen::MatrixXf MatrixXType;
typedef Eigen::VectorXf VectorXType;
typedef Eigen::RowVectorXf RowVectorXType;
typedef float STDType;
#endif // FLOAT_TYPE

#ifndef FLOAT_TYPE
typedef Eigen::MatrixXd MatrixXType;
typedef Eigen::VectorXd VectorXType;
typedef Eigen::RowVectorXd RowVectorXType;
typedef double STDType;
#endif


namespace Astral {
    class ActivationBase {
    public:
        virtual MatrixXType f(const MatrixXType& units, STDType a = 1) = 0;
        virtual MatrixXType df(const MatrixXType& units, STDType a = 1) = 0;
        virtual MatrixXType dLdz(const MatrixXType& z, const MatrixXType& dLdy) = 0;
    };

    class LossBase {
    public:
        virtual STDType f(const MatrixXType& input, const MatrixXType& target) = 0;
        virtual MatrixXType df(const MatrixXType& input, const MatrixXType& target) = 0;
    };

    class OptimizerBase {
    public:
        virtual void update(MatrixXType& params, const MatrixXType& grads) = 0;
    };

    class LayerBase {
    public:
        virtual void compile() = 0;

        virtual MatrixXType* forward(const MatrixXType& x) = 0;

        virtual MatrixXType backward(const MatrixXType& dLdy, OptimizerBase*) = 0;

        virtual ~LayerBase() { };
    };

    class Sigmoid : public ActivationBase {
    public:
        MatrixXType f(const MatrixXType& x, STDType a = 1) override {
            return x.unaryExpr([](STDType elem) -> STDType {
                return 1 / (1 + std::exp(-elem));
                });
        }

        MatrixXType df(const MatrixXType& x, STDType a = 1) override {
            return x.unaryExpr([](STDType elem) -> STDType {
                return std::exp(-elem) / std::pow(std::exp(-elem) + 1.0, 2);
                });
        }

        MatrixXType dLdz(const MatrixXType& z, const MatrixXType& dLdy) {
            MatrixXType dydz = df(z);
            return dydz.array() * dLdy.array();
        }
    };

    class Softmax : public ActivationBase {
    public:
        MatrixXType f(const MatrixXType& x, STDType a = 1) override {
            auto res = x.array().exp();
            VectorXType sumExp = res.rowwise().sum();
            return res.array().colwise() / sumExp.array();
        }

        MatrixXType df(const MatrixXType& units, STDType a = 1) override {
            MatrixXType Jacobi = -units.transpose() * units;
            for (int i = 0; i < units.cols(); ++i) Jacobi(i, i) += units(0, i); // Ö÷¶Ô½ÇÏß + pi
            return Jacobi;
        }

        MatrixXType dLdz(const MatrixXType& z, const MatrixXType& dLdy) {
            MatrixXType y = f(z);
            MatrixXType res(y.rows(), y.cols());
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
        MatrixXType f(const MatrixXType& units, STDType a = 1) override {
            return units.array().max(0.0);
        }
        MatrixXType df(const MatrixXType& units, STDType a = 1) override {
            return (units.array() >= 0.0).cast<STDType>();
        }
        MatrixXType dLdz(const MatrixXType& z, const MatrixXType& dLdy) override {
            MatrixXType y = f(z);
            MatrixXType dydz = df(y);
            return dydz.array() * dLdy.array();
        }
    };

    class LinearActivation : public ActivationBase {
    public:
        LinearActivation(STDType alpha) : alpha_(alpha) {};

        MatrixXType f(const MatrixXType& units, STDType a = 1) override {
            return alpha_ * units;
        }
        MatrixXType df(const MatrixXType& units, STDType a = 1) override {
            return MatrixXType::Constant(units.rows(), units.cols(), alpha_);
        }
        MatrixXType dLdz(const MatrixXType& z, const MatrixXType& dLdy) override {
            return dLdy;
        }

        STDType alpha_;
    };

    class MSELoss : public LossBase {
    public:
        STDType f(const MatrixXType& input, const MatrixXType& target) {
            auto res = (target - input).array().square();
            return res.sum() / res.size();
        }

        MatrixXType df(const MatrixXType& input, const MatrixXType& target) {
            return (target - input) * 2.0 / target.size();
        }
    };

    class CrossEntropyLoss : public LossBase {
    public:
        STDType f(const MatrixXType& input, const MatrixXType& target) {
            auto res = input.array().log() * target.array();
            return -res.sum() / input.rows();
        }

        MatrixXType df(const MatrixXType& input, const MatrixXType& target) {
            return -target.array() * input.array().inverse() / input.rows();
        }
    };

    class SGD : public OptimizerBase {
    public:
        STDType lr_;

        SGD(STDType lr) : lr_(lr) {};

        void setLr(STDType lr) { this->lr_ = lr; };

        void update(MatrixXType& params, const MatrixXType& grads) override {
            params -= lr_ * grads;
        }
    };

    class Linear : public LayerBase {
    public:
        Linear(int inFeatures, int outFeatures, ActivationBase* activation, bool bias = true);

        Linear(const Linear&) = default;

        Linear(Linear&&) = default;

        ~Linear() override;

        Linear& operator=(const Linear&) = default;

        Linear& operator=(Linear&&) = default;

        void compile() override;

        MatrixXType* forward(const MatrixXType& x) override;

        MatrixXType backward(const MatrixXType& dLdy, OptimizerBase*) override;

        const MatrixXType* x_;

        MatrixXType y_;

        MatrixXType z_;

        MatrixXType w_;

        MatrixXType b_;

        friend std::ostream& operator<<(std::ostream& os, const Linear& linear);

        int inFeatures_, outFeatures_;

        bool bias_;

        ActivationBase* activation_;

    }; // Layer

    class SelfAttention : public LayerBase {
    public:
        SelfAttention(int numVector, int embedSize, int heads);

        ~SelfAttention() override;

        void compile() override;

        MatrixXType* forward(const MatrixXType& x) override;

        MatrixXType backward(const MatrixXType& dLdy, OptimizerBase*) override;

        int numVector_, embedSize_, heads_;

        MatrixXType Wq_, Wk_, Wv_;

        MatrixXType Q_, K_, V_, Alpha_, Alpha2_;

        MatrixXType In_, Out_;
    };

    class Sequential {
    public:
        Sequential() = default;

        Sequential(const Sequential&) = default;

        Sequential(Sequential&&) = default;

        ~Sequential();

        Sequential& operator=(const Sequential&) = default;

        Sequential& operator=(Sequential&&) = default;

        void add(LayerBase* layer);

        void pop();

        void compile(OptimizerBase* optimizer, LossBase* lossfunc);

        void fit(const MatrixXType& x_train, const MatrixXType& y_train, int batchSize, int epochs);

        STDType evaluate(const MatrixXType& x_train, const MatrixXType& y_train);

        MatrixXType predict(const MatrixXType& x);

        MatrixXType forwardPropagation(const MatrixXType& input);

        STDType calculateLoss(const MatrixXType& input, const MatrixXType& target, LossBase* lossfunc);

        void backPropagation(const MatrixXType& forwardOutput, const MatrixXType& target);

        void log();

        std::vector<LayerBase*> layers_;

        OptimizerBase* optimizer_{};

        LossBase* lossfunc_{};
    }; // Seqiential

} // Astral