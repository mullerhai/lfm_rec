package recCore

import (
	"context"
	"fmt"
	"github.com/alexflint/go-memdump"
	"github.com/gogf/gf/container/gset"
	"github.com/rocketlaunchr/dataframe-go"
	"github.com/rocketlaunchr/dataframe-go/imports"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
	"os"
	"reflect"
	"sort"
)

type IdsMapping struct {
	UserIdIndexDict map[float64]float64
	IndexUserIdDict map[float64]float64
	ItemIdIndexDict map[float64]float64
	indexItemIdDict map[float64]float64
}

type LFM struct {
	classCount             int
	iterCount              int
	featureName            string
	labelName              string
	lr                     float64
	lam                    float64
	UserItemRatingMatrix   *mat.Dense
	ModelPfactor           *mat.Dense
	ModelQfactor           *mat.Dense
	RatingDf               *dataframe.DataFrame
	UseridItemidDict       map[float64]map[float64]float64
	UseridSet              *gset.Set
	ItemidSet              *gset.Set
	IdsMapping             *IdsMapping
	UserIndexItemIndexDict map[float64]map[float64]float64
}

//5 5 0.02 0.01 "userId" "itemId"
func NewDefaultLFM(ratingPath string, classCount, iterCount int, lr, lam float64, featureName, labelName string, featureArray []string) *LFM {
	userItemRatingMatrix, ratingDf := LoadData(ratingPath, featureArray)
	userIdItemIdDict := make(map[float64]map[float64]float64)
	userIndexItemIndexDict := make(map[float64]map[float64]float64)
	lfm := &LFM{classCount, iterCount, featureName, labelName, lr, lam, userItemRatingMatrix, nil, nil, ratingDf, userIdItemIdDict, nil, nil, nil, userIndexItemIndexDict}
	return lfm
}

func (lfm *LFM) generateUserIdItemIdIndexDict() {
	ratingDf := lfm.RatingDf
	useridIndexDict := lfm.IdsMapping.UserIdIndexDict
	itemidIndexDict := lfm.IdsMapping.ItemIdIndexDict
	iterator := ratingDf.Values(dataframe.ValuesOptions{0, 1, true}) // Don't apply read lock because we are write locking from outside.
	ratingDf.Lock()
	for {
		row, vals := iterator()
		if row == nil {
			break
		}
		userId, movieId, rating := vals["UserID"].(float64), vals["MovieID"].(float64), vals["Rating"].(float64)
		userIndex, movieIndex := float64(useridIndexDict[userId]), float64(itemidIndexDict[movieId])
		//fmt.Println("userid movieid ,rating ", userId, movieId, rating)
		//fmt.Println("userIndex  movieIndex ", userIndex, movieIndex)
		if _, ok := lfm.UseridItemidDict[userId]; ok {
			userMap := lfm.UseridItemidDict[userId]
			userIndexMap := lfm.UserIndexItemIndexDict[userIndex]
			if _, ok := userMap[movieId]; ok {
				ratingVal := userMap[movieId]
				newRatingVal := ratingVal + rating
				userMap[movieId] = newRatingVal
			} else {
				userMap[movieId] = rating
			}
			if _, ok := userIndexMap[movieIndex]; ok {
				ratingVals := userIndexMap[movieIndex]
				newRatingVals := ratingVals + rating
				userIndexMap[movieIndex] = newRatingVals
			} else {
				userIndexMap[movieIndex] = rating
			}
			lfm.UseridItemidDict[userId] = userMap
			lfm.UserIndexItemIndexDict[userIndex] = userIndexMap
		} else {
			movieRatingDict := map[float64]float64{movieId: rating}
			lfm.UseridItemidDict[userId] = movieRatingDict
			movieIndexRatingDict := map[float64]float64{movieIndex: rating}
			lfm.UserIndexItemIndexDict[userIndex] = movieIndexRatingDict
		}
		//fmt.Println(*row, vals)
	}
	ratingDf.Unlock()
	fmt.Println("lfm dict len ", len(lfm.UseridItemidDict))

}

func LoadData(ratingPath string, featureNameArr []string) (*mat.Dense, *dataframe.DataFrame) {
	var ctx = context.Background()
	file, err := os.Open(ratingPath)
	csvOp := imports.CSVLoadOptions{
		Comma:   ',',
		Comment: 0,
		DictateDataType: map[string]interface{}{
			//"UserID":    float64(0),
			//"MovieID":   float64(0),
			//"Rating":    float64(0),
			//"Timestamp": float64(0),
			featureNameArr[0]: float64(0),
			featureNameArr[1]: float64(0),
			featureNameArr[2]: float64(0),
			featureNameArr[3]: float64(0),
		},
	}
	fmt.Println(csvOp)
	ratingDf, err := imports.LoadFromCSV(ctx, file, csvOp)
	if err != nil {
		fmt.Println("load error ", err.Error())
	}
	//fmt.Println(RatingDf)
	float64Arr := SeriesConvertFloatArray(ratingDf)
	matrixRow := ratingDf.Series[0].NRows()
	matrixCol := len(ratingDf.Names())
	denseArr := mat.NewDense(matrixRow, matrixCol, float64Arr)

	return denseArr, ratingDf
}

func (lfm *LFM) InitModel() {
	src := rand.New(rand.NewSource(1))
	var uni = distuv.Uniform{0, 0.35, src}
	pDistinctEle := DistinctSeriesConvertSet(lfm.RatingDf.Series[0])
	qDistinctEle := DistinctSeriesConvertSet(lfm.RatingDf.Series[1])
	pClasslen := lfm.classCount * pDistinctEle.Size()
	qClasslen := lfm.classCount * qDistinctEle.Size()
	pData := GenerateUniformArr(uni, int64(pClasslen))
	qData := GenerateUniformArr(uni, int64(qClasslen))
	lfm.ItemidSet = &qDistinctEle
	lfm.UseridSet = &pDistinctEle
	lfm.ModelPfactor = mat.NewDense(pDistinctEle.Size(), lfm.classCount, pData)
	lfm.ModelQfactor = mat.NewDense(qDistinctEle.Size(), lfm.classCount, qData)
	lfm.generateIdsMatrixIndexMapping()
	lfm.generateUserIdItemIdIndexDict()

}

func (lfm *LFM) generateIdsMatrixIndexMapping() {
	var userIdArr = make([]float64, 0)
	var itemIdArr = make([]float64, 0)
	lfm.UseridSet.Iterator(func(v interface{}) bool {
		userIdArr = append(userIdArr, v.(float64))
		return true
	})
	lfm.ItemidSet.Iterator(func(v interface{}) bool {
		itemIdArr = append(itemIdArr, v.(float64))
		return true
	})
	var UserIdIndexDict = make(map[float64]float64)
	var IndexUserIdDict = make(map[float64]float64)
	var ItemIdIndexDict = make(map[float64]float64)
	var indexItemIdDict = make(map[float64]float64)
	for index, uid := range userIdArr {
		UserIdIndexDict[uid] = float64(index)
		IndexUserIdDict[float64(index)] = uid
	}
	for index, itemId := range itemIdArr {
		ItemIdIndexDict[itemId] = float64(index)
		indexItemIdDict[float64(index)] = itemId
	}
	idMapping := &IdsMapping{UserIdIndexDict, IndexUserIdDict, ItemIdIndexDict, indexItemIdDict}
	lfm.IdsMapping = idMapping
}

func DistinctSeriesConvertSet(series dataframe.Series) gset.Set {
	seriesArr := series.(*dataframe.SeriesFloat64).Values
	var newSet = gset.New(true)
	for _, ele := range seriesArr {
		newSet.Add(ele)
	}
	return *newSet
}

func (lfm *LFM) _preidct(userId, itemId float64) (float64, float64) {
	p := lfm.ModelPfactor.RowView(int(userId)).(*mat.VecDense)     //5,1
	q := lfm.ModelQfactor.RowView(int(itemId)).(*mat.VecDense).T() // 1.5
	var res mat.Dense
	res.Mul(p, q) //5,5
	//resRow, resCol := res.Caps()
	matR := mat.Sum(&res)
	logit := 1.0 / (1.0 + math.Exp(-matR))
	//fmt.Println("predict res shape  matR  logit ", resRow, resCol, matR, logit)
	return logit, matR
}

func (lfm *LFM) _loss(userIndex, itemIndex, rating float64) float64 {
	_, matR := lfm._preidct(userIndex, itemIndex)
	costError := rating - matR
	//fmt.Println("rating : {} ,cost error : {}, logit   ",rating , costError, logit)
	return costError
}

func (lfm *LFM) showShape(gradient_p, l2_p, pVal, qVal *mat.VecDense) {
	modp_r, modp_c := lfm.ModelPfactor.Caps()
	fmt.Println("ModelPfactor shape  ", modp_r, modp_c)
	modq_r, modq_c := lfm.ModelQfactor.Caps()
	fmt.Println("modelqfactor sh ape  ", modq_r, modq_c)
	g_r, g_c := gradient_p.Caps()
	p_r, p_c := pVal.Caps()
	q_r, q_c := qVal.Caps()
	fmt.Println("gradient_p shape  ", g_r, g_c)
	fmt.Println("pval shape ", p_r, p_c)
	fmt.Println("qval shape ", q_r, q_c)
	l2_p_r, l2_p_c := l2_p.Caps() //shape 5,1
	fmt.Println("l2_p shape ", l2_p_r, l2_p_c)
}
func (lfm *LFM) _optimize(userIndex, itemIndex, e float64) {
	pVal := lfm.ModelPfactor.RowView(int(userIndex)).(*mat.VecDense)
	qVal := lfm.ModelQfactor.RowView(int(itemIndex)).(*mat.VecDense)
	var gradient_p, l2_p, grad_l2_p, delta_p, gradient_q, l2_q, grad_l2_q, delta_q, np_val, nq_val mat.VecDense
	gradient_p.ScaleVec(-e, qVal)
	l2_p.ScaleVec(lfm.lam, pVal)
	grad_l2_p.AddVec(&gradient_p, &l2_p)
	delta_p.ScaleVec(lfm.lr, &grad_l2_p)
	gradient_q.ScaleVec(-e, pVal)
	l2_q.ScaleVec(lfm.lam, qVal)
	grad_l2_q.AddVec(&gradient_q, &l2_q)
	delta_q.ScaleVec(lfm.lr, &grad_l2_q)
	np_val.SubVec(pVal, &delta_p)
	nq_val.SubVec(qVal, &delta_q)
	lfm.ModelPfactor.SetRow(int(userIndex), vectofloat(&np_val))
	lfm.ModelQfactor.SetRow(int(itemIndex), vectofloat(&nq_val))
}

func vectofloat(vec mat.Vector) []float64 {
	vecArr := make([]float64, 0)
	newVec := vec.(*mat.VecDense)
	vecLen := newVec.Len()
	for i := 0; i < vecLen; i++ {
		newEle := newVec.AtVec(i)
		vecArr = append(vecArr, newEle)
	}
	return vecArr
}

func (lfm *LFM) Train(step int) {
	fmt.Println("lfm training ", lfm.iterCount)
	if step > lfm.iterCount {
		lfm.iterCount = step
	}
	for step := 0; step < lfm.iterCount; step++ {
		for userIdIndex, valDict := range lfm.UserIndexItemIndexDict {
			itemIdIndexs := reflect.ValueOf(valDict).MapKeys()
			//fmt.Println("step itemids len : ,userid ",step, len(itemIdIndexs), userIdIndex)
			for itemId := range itemIdIndexs {
				costErr := lfm._loss(userIdIndex, float64(itemId), valDict[float64(itemId)])
				lfm._optimize(userIdIndex, float64(itemId), costErr)
				//fmt.Println("step  costerror ", step, costErr)
			}
		}
		lfm.lr *= 0.9
	}
	fmt.Println("model training complete !!!!")
}

func (lfm *LFM) FromSeriesGenerateGSet(fieldName string) *gset.Set {
	itemIndex, _ := lfm.RatingDf.NameToColumn(fieldName)
	itemIds := lfm.RatingDf.Series[itemIndex].(*dataframe.SeriesFloat64).Values
	itemIdsInterfaceArr := FloatArrConvertInterfaceArray(itemIds)
	itemIdsSet := gset.New(true)
	itemIdsSet.Add(itemIdsInterfaceArr...)
	return itemIdsSet
}

func (lfm *LFM) Predict(userId float64, topN int32) PairList {
	fmt.Println("now predict userid ", userId)
	labelSet := gset.New(true)
	labelArr := DfWhere(lfm.RatingDf, lfm.featureName, lfm.labelName, userId)
	labelInterfaceArr := FloatArrConvertInterfaceArray(labelArr)
	labelSet.Add(labelInterfaceArr...)
	otherItemIds := lfm.ItemidSet.Diff(labelSet)
	//fmt.Println("otherItemIds  size :  label set size  ", otherItemIds.Size(), labelSet.Size())
	interestScoreDict := make(map[float64]float64, 0)
	userIndex := lfm.IdsMapping.UserIdIndexDict[userId]
	for _, itemId := range otherItemIds.Slice() {
		itemIndex := lfm.IdsMapping.ItemIdIndexDict[itemId.(float64)]
		_, score := lfm._preidct(userIndex, itemIndex)
		interestScoreDict[itemId.(float64)] = score
	}
	rankList := RankSortDict(interestScoreDict, topN)
	return rankList
}

type Pair struct {
	Key   float64
	Value float64
}

type PairList []Pair

func (p PairList) Len() int           { return len(p) }
func (p PairList) Less(i, j int) bool { return p[i].Value < p[j].Value }
func (p PairList) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

func RankSortDict(dict map[float64]float64, topN int32) PairList {
	pl := make(PairList, len(dict))
	i := 0
	for k, v := range dict {
		pl[i] = Pair{k, v}
		i++
	}
	sort.Sort(sort.Reverse(pl))
	topNRankPair := make(PairList, topN)
	for index, pair := range pl {
		//fmt.Println("sort pair ", pair.Key, pair.Value)
		topNRankPair[index] = Pair{pair.Key, pair.Value}
		if int32(index) == topN-1 {
			break
		}
	}
	return topNRankPair
}

func FloatArrConvertInterfaceArray(labelArr []float64) []interface{} {
	labelInterfaceArr := make([]interface{}, len(labelArr))
	for i, v := range labelArr {
		labelInterfaceArr[i] = v
	}
	return labelInterfaceArr
}

func DfWhere(df *dataframe.DataFrame, feature, label string, featureVal float64) []float64 {
	labelIndex, _ := df.NameToColumn(label)
	featureIndex, _ := df.NameToColumn(feature)
	labelArr := make([]float64, 0)
	labelSeries := df.Series[labelIndex].(*dataframe.SeriesFloat64)
	featureSeries := df.Series[featureIndex].(*dataframe.SeriesFloat64)
	for index, valz := range featureSeries.Values {
		if valz == featureVal {
			labelVal := labelSeries.Values[index]
			labelArr = append(labelArr, labelVal)
		}
	}
	return labelArr
}

func (lfm *LFM) DataFrameConvertDenseMatrix(df *dataframe.DataFrame) *mat.Dense {
	dataArr := SeriesConvertFloatArray(df)
	matrixRow := df.NRows()
	matrixCol := len(df.Names())
	userItemRatingMatrix := mat.NewDense(matrixRow, matrixCol, dataArr)
	return userItemRatingMatrix
}

func SeriesConvertFloatArray(df *dataframe.DataFrame) []float64 {
	series := df.Series
	arr := make([]float64, 0)
	for _, data := range series {
		temArr := data.(*dataframe.SeriesFloat64).Values
		arr = append(arr, temArr...)
	}
	return arr
}

func GenerateUniformArr(uni distuv.Uniform, arrLen int64) []float64 {
	x := make([]float64, arrLen)
	generateSample(x, uni)
	sort.Float64s(x)
	fmt.Println(x)
	return x
}

func generateSample(x []float64, r distuv.Rander) {
	for i := range x {
		x[i] = r.Rand()
	}
}

func (lfm *LFM) EvaluateLFMModel(selectCount, topN int) float64 {
	fmt.Println("begin EvaluateLFMModel  ")
	userIdSet := lfm.UseridSet
	userScoreSumDict := make(map[float64]float64, 0)
	for i := 1; i <= selectCount; i++ {
		scoreSum := float64(0)
		userId := userIdSet.Pop().(float64)
		//fmt.Println("pop from set userId ",userId)
		userIndex := lfm.IdsMapping.UserIdIndexDict[userId]
		itemIdScorePairList := lfm.Predict(userId, int32(topN))
		for _, itemScoreEle := range itemIdScorePairList {
			score := float64(itemScoreEle.Value)
			scoreSum += score
		}
		userIdSet.Add(userId)
		userScoreSumDict[userIndex] = scoreSum / float64(len(itemIdScorePairList))
		//fmt.Println("userIndex ,evalute sum  score ",userIndex,scoreSum)
	}
	selectUserScoreSum := float64(0)
	for _, scoreSum := range userScoreSumDict {
		selectUserScoreSum += scoreSum
	}
	avgScore := selectUserScoreSum / float64(len(userScoreSumDict))
	fmt.Println("evalute model avg score: ", avgScore)
	return avgScore
}

func (lfm *LFM) SaveModel(savePath string) {
	dumpPath := savePath
	w, err := os.Create(dumpPath)
	if err != nil {
		fmt.Println("has error")
	}
	memdump.Encode(w, &lfm)
}

func LoadModel(loadPath string) *LFM {
	r, err := os.Open(loadPath)
	if err != nil {
		fmt.Println("has error")
	}
	var lfmnew *LFM
	memdump.Decode(r, &lfmnew)
	return lfmnew
}
