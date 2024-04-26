#include "Astral.hpp"

namespace Astral {
    Softmax softmax;
    ReLU relu;
    MSELoss mseLoss;
    CrossEntropyLoss crossEntropyLoss;
    SGD sgd(0.01);
    Layer::Layer(int unitsNum, ActivationBase* activation, LayerType type) :
        unitsNum_(unitsNum), activation_(activation), type_(type) { }

    void Astral::Layer::log()
    {
        std::cout << "******** z ********" << std::endl;
        std::cout << this->z_ << std::endl;
        std::cout << "******** y ********" << std::endl;
        std::cout << this->y_ << std::endl;
        std::cout << "******** w ********" << std::endl;
        std::cout << this->w_ << std::endl;
        std::cout << "******** b ********" << std::endl;
        std::cout << this->b_ << std::endl;
    }

    void Astral::SelfAttention::compile()
    {
        this->Wq_ = Eigen::MatrixXd::Random(this->embedSize_, this->embedSize_);
        this->Wk_ = Eigen::MatrixXd::Random(this->embedSize_, this->embedSize_);
        this->Wv_ = Eigen::MatrixXd::Random(this->embedSize_, this->embedSize_);
    }

    Eigen::MatrixXd Astral::SelfAttention::forward(const Eigen::MatrixXd inputs)
    {
        Eigen::MatrixXd A; // embedSize_ * d
        if (inputs.cols() == this->embedSize_) A = inputs.transpose();
        else if (inputs.rows() == this->embedSize_) A = inputs;
        else throw std::runtime_error("Inputs' size does not match the config.");
        this->Q_ = this->Wq_ * A; // embedSize_ * d
        this->K_ = this->Wk_ * A;
        this->V_ = this->Wv_ * A;
        this->Alpha_ = this->Q_.transpose() * this->K_; // d * d
        this->Alpha2_ = Astral::softmax.f(this->Alpha_); // d * d
        return Alpha2_ * this->V_.transpose();
    }

    void Sequential::add(int unitsNum, ActivationBase* activation, LayerType type) {
        if (unitsNum < 0) throw std::invalid_argument("units must greater than 0");
        layers_.emplace_back(unitsNum, activation, type);
    }

    void Sequential::pop()
    {
        if (!layers_.empty())
            layers_.pop_back();
    }

    void Sequential::compile(OptimizerBase* optimizer, LossBase* loss)
    {
        if (!optimizer) throw std::runtime_error("optimizer can not be nullptr!");
        if (!loss) throw std::runtime_error("loss can not be nullptr!");
        this->lossfunc_ = loss;
        this->optimizer_ = optimizer;
        for (int i = 0; i < layers_.size(); ++i) {
            // 输入层没有偏置节点，权重矩阵依附在前一层，即输出层没有权重矩阵
            if (i != layers_.size() - 1)
                layers_[i].w_ = Eigen::MatrixXd::Random(layers_[i].unitsNum_, layers_[i + 1].unitsNum_);
            if (i != 0)
                layers_[i].b_ = Eigen::RowVectorXd::Random(layers_[i].unitsNum_);
        }
    }

    void Astral::Sequential::fit(const Eigen::MatrixXd& x_train,
        const Eigen::MatrixXd& y_train, int batchSize, int epochs)
    {
        if (batchSize < 0) throw std::invalid_argument("batchSize must greater than 0");
        if (epochs < 0) throw std::invalid_argument("epochs must greater than 0");
        for (int iteration = 1; iteration <= epochs; ++iteration) {
            int remainRows = x_train.rows(), curRow = 0, xcols = x_train.cols(), ycols = y_train.cols();
            while (remainRows > 0) {
                forwardPropagation(x_train.block(curRow, 0, std::min(remainRows, batchSize), xcols));
                double loss = calculateLoss(layers_.back().y_,
                    y_train.block(curRow, 0, std::min(remainRows, batchSize), ycols), lossfunc_);
                backPropagation(y_train.block(curRow, 0, std::min(remainRows, batchSize), ycols));
                std::cout << "iteration : " << iteration << ", loss = " << loss << std::endl;
                remainRows -= batchSize, curRow += batchSize;
                // log();
            }
        }
    }

    void Astral::Sequential::forwardPropagation(const Eigen::MatrixXd& input)
    {
        layers_[0].y_ = input;
        for (int i = 1; i < layers_.size(); ++i) {
            const auto& lastLayer = layers_[i - 1];
            auto& layer = layers_[i];
            layer.z_ = lastLayer.y_ * lastLayer.w_;
            Eigen::RowVectorXd b = layer.b_;
            layer.z_ = layer.z_.rowwise() + b; // Eigen广播机制
            // std::cout << layer.z_ << std::endl;
            layer.y_ = layer.activation_->f(layer.z_, 0);
            // std::cout << layer.y_ << std::endl;
        }
    }

    double Astral::Sequential::calculateLoss(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target, LossBase* loss)
    {
        return loss->f(input, target);
    }

    void Astral::Sequential::backPropagation(const Eigen::MatrixXd& target)
    {
        auto dLdy = this->lossfunc_->df(layers_.back().y_, target);
        int layersNum = layers_.size();
        for (int layerIndex = layersNum - 1; layerIndex > 0; --layerIndex) {
            auto& foreLayer = layers_[layerIndex - 1];
            auto& curLayer = layers_[layerIndex];
            // curLayer.activation_->df(curLayer.z_);
            Eigen::MatrixXd dLdz = curLayer.activation_->dLdz(curLayer.z_, dLdy); // batchsize * curLayer.unitsNum
            // std::cout << dLdz << std::endl;
            Eigen::MatrixXd wgrads = foreLayer.y_.transpose() * dLdz;
            // std::cout << grads << std::endl;
            dLdy = dLdz * foreLayer.w_.transpose(); // (w * dLdz^T)^T = dLdz * w^T
            // std::cout << dLdy << std::endl;
            Eigen::MatrixXd bgrads = dLdz.colwise().sum();
            // std::cout << "layer " << layerIndex << " weights' grads\n" << wgrads << std::endl;
            // std::cout << "layer " << layerIndex + 1 << " bias' grads\n" << bgrads << std::endl;
            optimizer_->update(curLayer.b_, bgrads); // target.rows() == batchsize
            optimizer_->update(foreLayer.w_, wgrads);
        }
    }

    double Astral::Sequential::evaluate(const Eigen::MatrixXd& x_train, const Eigen::MatrixXd& y_train)
    {
        forwardPropagation(x_train);
        double loss = calculateLoss(layers_.back().y_, y_train, lossfunc_);
        return loss;
    }

    Eigen::MatrixXd Astral::Sequential::predict(const Eigen::MatrixXd& x)
    {
        forwardPropagation(x);
        return layers_.back().y_;
    }

    void Astral::Sequential::log()
    {
        int cnt = 1;
        for (auto& layer : layers_) {
            std::cout << "Layer " << cnt++ << " info : " << std::endl;
            layer.log();
            std::cout << std::endl;
        }
    }
    

}