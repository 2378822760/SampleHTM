#include "Astral.hpp"
#define DEBUG
namespace Astral {
    Softmax softmax;
    ReLU relu;
    LinearActivation linearActivation(1.0);
    MSELoss mseLoss;
    CrossEntropyLoss crossEntropyLoss;
    SGD sgd(0.01);

    Sequential::~Sequential()
    {
        this->optimizer_ = nullptr;
        this->lossfunc_ = nullptr;
        for (auto& layer : layers_) {
            delete layer;
            layer = nullptr;
        }
    }

    void Sequential::add(LayerBase* layer) {

        layers_.push_back(layer);
    }

    void Sequential::pop()
    {
        if (!layers_.empty()) {
            delete layers_.back();
            layers_.pop_back();
        }
    }

    void Sequential::compile(OptimizerBase* optimizer, LossBase* loss)
    {
        if (!optimizer) throw std::runtime_error("optimizer can not be nullptr!");
        if (!loss) throw std::runtime_error("loss can not be nullptr!");
        this->lossfunc_ = loss;
        this->optimizer_ = optimizer;
        for (auto& layer : this->layers_) layer->compile();
    }

    void Sequential::fit(const MatrixXType& x_train,
        const MatrixXType& y_train, int batchSize, int epochs)
    {
        if (batchSize < 0) throw std::invalid_argument("batchSize must greater than 0");
        if (epochs < 0) throw std::invalid_argument("epochs must greater than 0");
        for (int iteration = 1; iteration <= epochs; ++iteration) {
            int remainRows = x_train.rows(), curRow = 0, xcols = x_train.cols(), ycols = y_train.cols();
            while (remainRows > 0) {
                int blockRows = std::min(remainRows, batchSize);
                MatrixXType res = forwardPropagation(x_train.block(curRow, 0, blockRows, xcols));
                STDType loss = calculateLoss(res, y_train.block(curRow, 0, blockRows, ycols), lossfunc_);
                backPropagation(res, y_train.block(curRow, 0, blockRows, ycols));
                std::cout << "iteration : " << iteration << ", loss = " << loss << std::endl;
                remainRows -= batchSize, curRow += batchSize;
                // log();
            }
        }
    }

    MatrixXType Sequential::forwardPropagation(const MatrixXType& input)
    {
        if (this->layers_.empty()) std::cout << "Warning: this network is empty." << std::endl;
        const MatrixXType* lastLayerOutput = new MatrixXType(input);
        for (auto& layer : layers_) {
            lastLayerOutput = layer->forward(*lastLayerOutput);
        }
        return *lastLayerOutput; // copy
    }

    STDType Sequential::calculateLoss(const MatrixXType& input, const MatrixXType& target, LossBase* loss)
    {
        return loss->f(input, target);
    }

    void Sequential::backPropagation(const MatrixXType& forwardOutput, const MatrixXType& target)
    {
        auto dLdy = this->lossfunc_->df(forwardOutput, target);
        for (auto layer = this->layers_.rbegin(); layer != this->layers_.rend(); ++layer) {
            dLdy = (*layer)->backward(dLdy, this->optimizer_);
        }
    }

    STDType Sequential::evaluate(const MatrixXType& x_train, const MatrixXType& y_train)
    {
        auto res = forwardPropagation(x_train);
        STDType loss = calculateLoss(res, y_train, lossfunc_);
        return loss;
    }

    MatrixXType Sequential::predict(const MatrixXType& x)
    {
        return forwardPropagation(x);
    }

    void Sequential::log()
    {
        int cnt = 1;
        for (auto& layer : layers_) {
            std::cout << "Layer " << cnt++ << " info : " << std::endl;
            std::cout << layer << std::endl;
        }
    }

    std::ostream& operator<<(std::ostream& os, const Linear& linear)
    {
        os << "******** z ********" << '\n';
        os << linear.z_ << '\n';
        os << "******** y ********" << '\n';
        os << linear.y_ << '\n';
        os << "******** w ********" << '\n';
        os << linear.w_ << '\n';
        os << "******** b ********" << '\n';
        os << linear.b_ << '\n';
        return os;
    }

    Linear::Linear(int inFeatures, int outFeatures, ActivationBase* activation, bool bias)
        : bias_(bias), activation_(activation)
    {
        assert(inFeatures > 0);
        assert(outFeatures > 0);
        this->x_ = nullptr;
        this->inFeatures_ = inFeatures;
        this->outFeatures_ = outFeatures;
    }

    Linear::~Linear()
    {
        this->x_ = nullptr;
        this->activation_ = nullptr;
    }

    void Linear::compile()
    {
        this->w_ = MatrixXType::Random(this->inFeatures_, this->outFeatures_);
        if (bias_) this->b_ = MatrixXType::Random(1, this->outFeatures_);
    }

    MatrixXType* Linear::forward(const MatrixXType& x)
    {
        this->x_ = &x;
        this->z_ = x * this->w_;
        if (this->bias_) {
            RowVectorXType b = this->b_;
            this->z_ = this->z_.rowwise() + b; // Eigen�㲥����
        }
        // std::cout << layer.z_ << std::endl;
        this->y_ = this->activation_->f(this->z_, 0);
        return &this->y_;
    }

    MatrixXType Linear::backward(const MatrixXType& dLdy, OptimizerBase* optimizer)
    {
        MatrixXType dLdz = this->activation_->dLdz(this->z_, dLdy); // batchsize * curLayer.unitsNum
        MatrixXType wgrads = (*this->x_).transpose() * dLdz;
        this->x_ = nullptr;
        optimizer->update(this->w_, wgrads);
        if (this->bias_) {
            MatrixXType bgrads = dLdz.colwise().sum();
            optimizer->update(this->b_, bgrads); // target.rows() == batchsize
        }
        return dLdz * this->w_.transpose(); // dLdy = (w * dLdz^T)^T = dLdz * w^T
    }

    SelfAttention::SelfAttention(int numVector, int embedSize, int heads)
    {
        assert(numVector > 0);
        assert(embedSize > 0);
        assert(heads > 0);
        this->numVector_ = numVector;
        this->embedSize_ = numVector;
        this->heads_ = heads;
    }

    SelfAttention::~SelfAttention()
    {
    }

    void SelfAttention::compile()
    {
        this->Wq_ = MatrixXType::Random(this->embedSize_, this->embedSize_);
        this->Wk_ = MatrixXType::Random(this->embedSize_, this->embedSize_);
        this->Wv_ = MatrixXType::Random(this->embedSize_, this->embedSize_);
    }
    MatrixXType* SelfAttention::forward(const MatrixXType& x)
    {
        this->In_ = x;
        this->Q_ = In_ * this->Wq_; // numVector * embedSize
        this->K_ = In_ * this->Wk_;
        this->V_ = In_ * this->Wv_;
        this->Alpha_ = this->Q_ * this->K_.transpose(); // d * d
        this->Alpha2_ = Astral::softmax.f(this->Alpha_); // d * d 
        this->Out_ = this->Alpha2_ * this->V_;
        return &this->Out_;
    }

    MatrixXType SelfAttention::backward(const MatrixXType& dLdO, OptimizerBase* optimizer)
    {
        auto dLdV = this->Alpha2_.transpose() * dLdO;
        auto dLdWv = this->In_.transpose() * dLdV;
        auto dLdA2 = dLdO * this->V_.transpose();
        auto dLdA = Astral::softmax.dLdz(this->Alpha_, dLdA2);
        auto dLdKT = this->Q_.transpose() * dLdA;
        auto dLdK = dLdKT.transpose();
        auto dLdQ = dLdA * this->K_;
        auto dLdWk = this->In_.transpose() * dLdK;
        auto dLdWq = this->In_.transpose() * dLdQ;
        auto dLdIn = dLdQ * this->Wq_.transpose() + dLdK * this->Wk_.transpose() + dLdV * this->Wv_.transpose();
#ifdef DEBUG
        std::cout << "--         grad debug         --\n";
        std::cout << dLdV << '\n';
        std::cout << dLdWv << '\n';
        std::cout << dLdA2 << '\n';
        std::cout << dLdA << '\n';
        std::cout << dLdK << '\n';
        std::cout << dLdQ << '\n';
        std::cout << dLdWk << '\n';
        std::cout << dLdWq << '\n';
        std::cout << dLdIn << '\n';

#endif // DEBUG
        optimizer->update(this->Wq_, dLdWq);
        optimizer->update(this->Wk_, dLdWk);
        optimizer->update(this->Wv_, dLdWv);
        return dLdIn;
    }
}