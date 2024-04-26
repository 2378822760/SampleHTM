#pragma once
#include <vector>
#include <map>
#include "type.hpp"
/**
 * @brief 新皮质中的突触的数据结构,一个突触连接一个column和输入中的一个bit。
 * 包含了永久值permanence和连接的输入空间索引
 */
struct synapse
{
    synapse() {
        this->permanence = 0.0;
        this->sourceInputIndex = 0;
    }
    synapse(Real sourceInputIndex, UInt permanence = 0.0) {
        this->permanence = permanence;
        this->sourceInputIndex = sourceInputIndex;
    }
    Real permanence;
    UInt sourceInputIndex;
};

using namespace std;


class SpatialPooler {
public:
    /** 空间池的构造函数
     * @brief 空间池所需要的数据结构分配空间以及随机初始化权值
     * @param inputDimensions 输入维度。用于计算输入的比特数
     * @param columnDimensions column维度。用于计算空间池column数量
     * @param potentialRadius 每个column的感受野半径，感受野大小为
     *      2 * potentialRadius + 1
     * @param potentialPct 每个column可以连接感受野内输入的比例。通过计算
     *      我们的得到一个column的隐含突触数量
     *      (2 * potentialRadius + 1) * potentialPct
     * @param globalInhibition 是否开启全局抑制。全局抑制意味着每个column
     *      的竞争对象为所有column。
     * @param localAreaDensity 在一个抑制半径内可以获胜的column的比例。该
     *      参数和numActiveColumnsPerInhArea只可以选择一个。若要使用该参数
     *      numActiveColumnsPerInhArea应为负数。
     * @param numActiveColumnsPerInhArea 在一个抑制半径内可以获胜的column
     *      的数量。该参数和localAreaDensity只可以选择一个。若要使用该参数
     *      localAreaDensity应为负数。
     * @param stimulusThreshold 刺激阈值。参与竞争的column的overlap值大于
     *      该阈值才可能被激活。
     * @param synPermInactiveDec 针对被激活的column，若其突触未连接，那么
     *      该突触的永久值 - synPermInactiveDec
     * @param synPermActiveInc 针对被激活的column，若其突触连接，那么
     *      该突触的永久值 + synPermActiveInc
     * @param synPermConnected 突触激活阈值。当突触的永久值大于该阈值时，
     *      该突触被激活。
     * @param minPctOverlapDutyCycles 用于更新minOverlapDutyCycles。
     * @param dutyCyclePeriod 更新dutyCycles时的迭代次数收敛于该参数。换而
     *      言之，如果当前迭代次数少于该值，那么计算dutyCycles时就除以迭代
     *      次数。当迭代次数大于该参数时，除以该参数。
     * @param boostStrength 更新boostFactors时，boostFunction使用的参数β。
     */
    SpatialPooler(vector<UInt> inputDimensions, vector<UInt> columnDimensions,
        UInt potentialRadius = 3, Real potentialPct = 0.5,
        bool globalInhibition = true, Real localAreaDensity = -1.0,
        UInt numActiveColumnsPerInhArea = 10,
        UInt stimulusThreshold = 0, Real synPermInactiveDec = 0.008,
        Real synPermActiveInc = 0.05, Real synPermConnected = 0.1,
        Real minPctOverlapDutyCycles = 0.001,
        UInt dutyCyclePeriod = 1000, Real boostStrength = 0.0);

    /**
     * @brief 使用配置文件构造空间池
     * @param config 配置
    */
    SpatialPooler(const map<string, string>& config);

    ~SpatialPooler() = default;

    void init(vector<UInt> inputDimensions, vector<UInt> columnDimensions,
        UInt potentialRadius = 3, Real potentialPct = 0.5,
        bool globalInhibition = true, Real localAreaDensity = -1.0,
        UInt numActiveColumnsPerInhArea = 10,
        UInt stimulusThreshold = 0, Real synPermInactiveDec = 0.008,
        Real synPermActiveInc = 0.05, Real synPermConnected = 0.1,
        Real minPctOverlapDutyCycles = 0.001,
        UInt dutyCyclePeriod = 1000, Real boostStrength = 0.0);

    /**
     * @brief 空间池最主要的函数。根据传递给函数的输入，计算出对应的激活column。
     * 开启学习后会进行参数更新。
     * @param inputVector 二值(0,1)向量。该向量通过编码器获得。向量长度应严
     *      格等于numInputs。
     * @param learn 如果开启学习，那么SP中一些参数会进行更新。
     * @param activeVector 二值(0,1)向量，若该值为1，那么说明该column被激活。
     *      该向量作为本次输入的输出。其大小为numColumns。
     * @return 结果保存在activeVector中
     */
    vector<UInt> compute(const vector<UInt>& inputVector, bool learn);

    /**
     * @return 空间池(SP)中columns的数量
     */
    UInt columnsNum() { return this->numColumns_; };

    UInt inputsNum() { return this->numInputs_; };

    void printParameters() const;
private:
    /**
     * @brief 为column映射感受野，
     * @param column
     * @return
    */
    UInt mapColumn_(UInt column);

    vector<UInt> mapPotential_(UInt column, bool wrapAround);
    /**
     * @brief 为每一个源索引映射一组目标索引（mapRadius为映射半径）。
     * 将每组索引存放到二维数组container。存放之前会先对该容器进行clear操作。
     * 该算法为每个源索引分配的目标索引是连续的。
     * @param sourceSize 需要映射的源索引数量
     * @param targetSize 需要映射的目标索引数量
     * @param mapRadius 每个源索引的映射半径,总共映射的数量为(2 * mapRadius + 1)
     * @param container 映射结果以下标对保存在该参数中。
     * 其中每个一维向量的内容为{beg1, end1,(beg2, end2)}。最多包含两组索引
     * 范围，有第二组是因为边界元素索引为中心所以会超过左边界从右边界索引。
     */
    void mapIndex_(const UInt sourceSize, const UInt targetSize, UInt mapRadius,
        vector<vector<UInt>>& container);

    void mapIndexE_(const UInt beg, const UInt end, const UInt size,
        const UInt mapRadius, vector<vector<UInt>>& container);

    /**
     * @brief （初始化时）为每个column分配突触。给定一个感受野范围（由mapIndex获得）
     * 根据参数potentialPct为感受野内输入随机分配突触，并随机赋初始值。
     * 暂时未考虑到论文中提到的天然中心。
     * @param receptiveField 一维向量的内容为{beg1, end1,(beg2, end2)}，
     *      最多包含两组索引范围。
     */
    void assignSynapses_(vector<UInt>& receptiveField);

    /**
     * @brief 更新半径。如果开启全局抑制，那么抑制范围就为numColumns。否则抑制半径
     * 需要更新为每个column连接突触个数的平均值。
     */
    void updateInhibitionRadius_();

    /**
     * @brief 为每个输入计算overlap。每个column的overlap值为connectedSyn连接到为
     * 1的输入的数量。
     * @param inputVector 输入向量(二值0，1)，通过encoder获得。
     */
    void calculateOverlap_(const vector<UInt>& inputVector);

    /**
     * @brief 开启学习后，每个column的overlap乘一个boostFactor，来作为最终的overlap。
     */
    void boostOverlap_();

    /**
     * @brief 抑制。在抑制半径范围内选取激活（胜利）的column。如果开启了全局抑制，
     * 选举范围为所有columns。
     */
    vector<UInt> inhibitColumns_();

    /**
     * @brief 返回第columnIndex个column邻居中第k大的overlap值，作为激活时的依据。
     */
    Real kthScore(UInt columnIndex, UInt numActive);

    /**
     * @brief 更新每个column的邻居。在更新抑制半径后需要调用此函数，显式的更新每
     * 个column的邻居。
     */
    void updateNeighbour_();

    /**
     * @brief 调整所有激活column的突触永久值。若突触连接增加永久值，反之减少。
     */
    void adaptSynapses_();

    /**
     * @brief 更新activeDutyCycle_和overlapDutyCycle_。
     */
    void updateDutyCycles_(const vector<UInt>& activeVector);

    /**
     * @brief 更新所有的BoostFactor。
     */
    void updateBoostFactors_();

    /**
     * @brief 如果overlapDutyCycle(c) < minOverlapDutyCycle(c)那么需要增加该
     * column所有突触的永久值。
     */
    void bumpUpWeakColumns_();

    /**
     * @brief 更新每个column的minOverlapDutyCycle。
     */
    void updateMinDutyCycles_();

    /**
     * @brief 为第colIndex个column所有的突触的永久值增加incFactor。
     * @param colIndex column的索引。
     * @param incFactor 增涨幅度。permenence += incFactor
     */
    void increasePermanences_(UInt colIndex, Real incFactor);
private:
    UInt numInputs_; // 输入的一维向量的长度
    UInt numColumns_; // SP层中columns的数量
    vector<UInt> columnDimensions_; // column的维度，用于计算numColumns_
    vector<UInt> inputDimensions_; // 输入的维度，用于计算numInputs_
    UInt potentialRadius_;  //每个column的感受野半径，感受野大小为(2 * potentialRadius + 1)
    /*
    The percent of the inputs, within a column's potential radius, that are initialized to be in
    this column’s potential synapses. This should be set so that on average, at least 15-20
    input bits are connected when the Spatial Pooling algorithm is initialized. For example,
    suppose the input to a column typically contains 40 ON bits and that permanences are
    such that 50% of the synapses are initially connected. In this case you will want
    potentialPct to be at least 0.75 since 40*0.5*0.75 = 15.
    */
    Real potentialPct_;
    Real initConnectedPct_;  // 初始化时连接的比例
    bool globalInhibition_;  // 是否开启全局抑制全局抑制意味着每个column的竞争对象为所有columns。

    int numActiveColumnsPerInhArea_; // 在一个抑制半径内可以获胜的column的数量。该参数和localAreaDensity只可以选择一个。
    /* 在一个抑制半径内可以获胜的column的比例。该参数和numActiveColumnsPerInhArea只可以选择一个。
    若要使用该参数numActiveColumnsPerInhArea应为负数。
     */
    Real localAreaDensity_;
    UInt stimulusThreshold_;  // 刺激阈值。参与竞争的column的overlap值大于该阈值才可能被激活。
    Real synPermInactiveDec_;  // 针对被激活的column，若其突触未连接，那么该突触的永久值 - synPermInactiveDec
    Real synPermActiveInc_;  // 针对被激活的column，若其突触连接，那么该突触的永久值 + synPermActiveInc
    Real synPermConnected_;  // 突触激活阈值。当突触的永久值大于该阈值时，该突触被激活。
    Real minPctOverlapDutyCycles_;  // 用于更新minOverlapDutyCycles。
    UInt inhibitionRadius_;  // 抑制半径
    UInt dutyCyclePeriod_;  // 更新dutyCycles时的迭代次数收敛于该参数。换而言之，如果当前迭代次数少于该值，那么计算dutyCycles时就除以迭代次数。当迭代次数大于该参数时，除以该参数。
    Real boostStrength_;  // 更新boostFactors时，boostFunction使用的参数β。
    UInt iterationNum_; // 当前迭代次数
    UInt updatePeriod_; // 更新周期

    vector<vector<UInt>> potentialPools_; // 每个column的感受野
    vector<vector<synapse>> potentialSyn_; // 每个column的潜在连接
    vector<vector<synapse>> connectedSyn_; // 每个column的连接

    vector<vector<UInt>> neighbours_; // 每个column的邻居
    vector<UInt> overlap_;
    vector<Real> boostFactors_;
    vector<Real> boostedOverlaps_;
    vector<UInt> activeColumns_;

    vector<Real> activeDutyCycle_;
    vector<Real> overlapDutyCycle_;
    vector<Real> minOverlapDutyCycles_;
    // vector<UInt> activeCount_;
    // vector<UInt> overlapCount_;
};