#pragma once
#include "type.hpp"
#include <vector>
#include <unordered_set>
#include <map>

using namespace std;

struct Synapse {
    CellIndex presynapticCell;
    Permanence permanence;
};

struct Segment {
    CellIndex cell;             // 保存该seg所依附的细胞
    vector<Synapse> synapses;   // syn不能单独存在，要依附于seg
    UInt numActivePotentialSynapses;
};

struct Column {
    vector<SegIndex> activeSegments;   // t-1时刻，每当一次计算后由activeSegments更新
    vector<SegIndex> matchingSegments; // t-1时刻，每当一次计算后由matchingSegments更新
    vector<CellIndex> cells;
};

class TemporalMemory {
private:
    /* 全局变量 */
    UInt ACTIVATION_THRESHOLD{};
    UInt LEARNING_THRESHOLD{};
    UInt SYNAPSE_SAMPLE_SIZE{};
    Permanence PERMANENCE_INCREMENT{};
    Permanence PERMANENCE_DECREMENT{};
    Permanence CONNECTED_PERMANENCE{};
    Permanence PREDICTED_DECREMENT{};
    Permanence INITIAL_PERMANENCE{};

    /* 过程变量 */
    bool LEARNING_ENABLED{};
    UInt iteration_{};

    /* 初始化变量，在构造函数或者调用initialize函数中初始化 */
    vector<UInt> columnDimensions_;
    UInt cellsPerColumn_{};
    UInt numColumns_{};

    vector<Column> columns;
    vector<ColumnIndex> cellsFa;    // 纽带，保存每个cell所依附的col
    unordered_set<CellIndex> candidates;   // t-1时刻，每当一次计算后由winnerCells更新
    vector<bool> cells;             // t-1时刻，每当一次计算后由activeCells更新
    vector<bool> winnerCells;       // t时刻
    vector<bool> activeCells;       // t时刻
    vector<UInt> cellsNumSegments;  // 每个cell上segment的数量。全局共享，无需区分时刻

    vector<Segment> segments;
    vector<SegIndex> activeSegments;   // t时刻
    vector<SegIndex> matchingSegments; // t时刻
public:
    TemporalMemory();
    /**
     * @brief 时间池(TM)的构造函数
     * @param columnDimensions columns的维度，用于计算columns的数量。
     * @param cellsPerColumn 每个column上细胞的数量。
     * @param activationThreshold segment的激活阈值，如果一个segment上处于激活状态的
     * synapses数量≥该阈值，那么该segment就会处于激活状态。
     * @param initialPermanence synapse的初始permanence。
     * @param connectedPermanence synapse的激活阈值，如果一个synapse的permanence≥该阈值
     * 那么该synapse就会处于激活状态。
     * @param LearningThreshold segment的学习阈值，如果一个segment中激活突触的数量≥该值，
     * 则该segment处于matching状态，有资格grow和强化与先前活动细胞的突触。
     * @param permanenceIncrement 如果一个segment正确地预测了细胞的活动，那么其激活突触的
     * permanence就会增加这个量。
     * @param permanenceDecrement 如果一个segment正确地预测了细胞的活动，那么其非激活突触
     * 的permanence就会减少这个量。
     * @param predictedSegmentDecrement 如果一个segment错误地预测了细胞的活动，那么其激活
     * 突触的permanence就会减少这个量。
    */
    TemporalMemory(const vector<UInt>& columnDimensions, UInt cellsPerColumn = 32,
        UInt activationThreshold = 13, Permanence initialPermanence = 0.21,
        Permanence connectedPermanence = 0.50, UInt LearningThreshold = 10,
        Permanence permanenceIncrement = 0.10, Permanence permanenceDecrement = 0.10,
        Permanence predictedSegmentDecrement = 0.0);

    TemporalMemory(const map<string, string>& config);

    ~TemporalMemory();

    void initialize(const vector<UInt>& columnDimensions, UInt cellsPerColumn = 32,
        UInt activationThreshold = 13, Permanence initialPermanence = 0.21,
        Permanence connectedPermanence = 0.50, UInt LearningThreshold = 10,
        Permanence permanenceIncrement = 0.10, Permanence permanenceDecrement = 0.10,
        Permanence predictedSegmentDecrement = 0.0);

    /**
     * @brief 
     * @param activeColumnsSize 
     * @param activeColumns 
     * @param learn 
     * @return 
    */
    vector<CellIndex> compute(size_t activeColumnsSize, const UInt *activeColumns,
        bool learn = true);
    
    UInt numberOfCells();

private:
    void activateCells(size_t activeColumnsSize, const UInt activeColumns[]);

    void activatePredictedColumn(Column& column);

    void burstColumn(Column& column);

    void punishPredictedColumn(Column& column);

    void activateDendrites();

    /* Helper Function */

    SegIndex growNewSegment(CellIndex cell);

    CellIndex leastUsedCell(Column& column);

    SegIndex bestMatchingSegment(Column& column);

    void growSynapses(SegIndex segment, UInt newSynapseCount);

    template<class T>
    T chooseRandom(vector<T>& Candidates);

    template<class T>
    T chooseRandom(unordered_set<T>& Candidates);

    void createNewSynapse(SegIndex index_seg, CellIndex presynapticCell);

    void updateSegments();

    void updateCells();
};