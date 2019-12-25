package recCore

import (
	"context"
	"fmt"
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

//func main() {
//	ratingPath := "./recCore/ratings.csv"
//	lfm := NewDefaultLFM(ratingPath)
//	lfm.InitModel()
//	lfm.Train()
//	userId := 23.0
//	var topN int32 = 10
//	var itemScore = lfm.Predict(userId, topN)
//	for index, itemScoreEle := range itemScore {
//		itemId := float64(itemScoreEle.Key)
//		fmt.Println("item : %f ,score : %f ", itemId, itemScoreEle.Value)
//		if index == int(topN)*50 {
//			break
//		}
//	}
//}

type IdsMapping struct {
	userIdIndexDict map[float64]float64
	indexUserIdDict map[float64]float64
	itemIdIndexDict map[float64]float64
	indexItemIdDict map[float64]float64
}

type LFM struct {
	classCount             int
	iterCount              int
	featureName            string
	labelName              string
	lr                     float64
	lam                    float64
	userItemRatingMatrix   *mat.Dense
	modelPfactor           *mat.Dense
	modelQfactor           *mat.Dense
	ratingDf               *dataframe.DataFrame
	useridItemidDict       map[float64]map[float64]float64
	useridSet              *gset.Set
	itemidSet              *gset.Set
	idsMapping             *IdsMapping
	userIndexItemIndexDict map[float64]map[float64]float64
}

func NewDefaultLFM(ratingPath string) *LFM {
	userItemRatingMatrix, ratingDf := loadData(ratingPath)
	userIdItemIdDict := make(map[float64]map[float64]float64)
	userIndexItemIndexDict := make(map[float64]map[float64]float64)
	lfm := &LFM{5, 5, "userId", "itemId", 0.02, 0.01, userItemRatingMatrix, nil, nil, ratingDf, userIdItemIdDict, nil, nil, nil, userIndexItemIndexDict}
	return lfm
}

func (lfm *LFM) generateUserIdItemIdIndexDict() {
	ratingDf := lfm.ratingDf
	useridIndexDict := lfm.idsMapping.userIdIndexDict
	itemidIndexDict := lfm.idsMapping.itemIdIndexDict
	iterator := ratingDf.Values(dataframe.ValuesOptions{0, 1, true}) // Don't apply read lock because we are write locking from outside.
	ratingDf.Lock()
	for {
		row, vals := iterator()
		if row == nil {
			break
		}
		userId, movieId, rating := vals["UserID"].(float64), vals["MovieID"].(float64), vals["Rating"].(float64)
		userIndex, movieIndex := float64(useridIndexDict[userId]), float64(itemidIndexDict[movieId])
		fmt.Println("userid movieid ,rating ", userId, movieId, rating)
		fmt.Println("userIndex  movieIndex ", userIndex, movieIndex)
		if _, ok := lfm.useridItemidDict[userId]; ok {
			userMap := lfm.useridItemidDict[userId]
			userIndexMap := lfm.userIndexItemIndexDict[userIndex]
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
			lfm.useridItemidDict[userId] = userMap
			lfm.userIndexItemIndexDict[userIndex] = userIndexMap
		} else {
			movieRatingDict := map[float64]float64{movieId: rating}
			lfm.useridItemidDict[userId] = movieRatingDict
			movieIndexRatingDict := map[float64]float64{movieIndex: rating}
			lfm.userIndexItemIndexDict[userIndex] = movieIndexRatingDict
		}
		//fmt.Println(*row, vals)
	}
	ratingDf.Unlock()
	fmt.Println("lfm dict len ", len(lfm.useridItemidDict))

}

func loadData(ratingPath string) (*mat.Dense, *dataframe.DataFrame) {
	var ctx = context.Background()
	file, err := os.Open(ratingPath)
	csvOp := imports.CSVLoadOptions{
		Comma:   ',',
		Comment: 0,
		DictateDataType: map[string]interface{}{
			"UserID":    float64(0),
			"MovieID":   float64(0),
			"Rating":    float64(0),
			"Timestamp": float64(0),
		},
	}
	fmt.Println(csvOp)
	ratingDf, err := imports.LoadFromCSV(ctx, file, csvOp)
	if err != nil {
		fmt.Println("load error ", err.Error())
	}
	//fmt.Println(ratingDf)
	float64Arr := seriesConvertFloatarray(ratingDf)
	matrixRow := ratingDf.Series[0].NRows()
	matrixCol := len(ratingDf.Names())
	denseArr := mat.NewDense(matrixRow, matrixCol, float64Arr)

	return denseArr, ratingDf
}

func (lfm *LFM) InitModel() {
	src := rand.New(rand.NewSource(1))
	var uni = distuv.Uniform{0, 0.35, src}
	pDistinctEle := distinctSeriesConvertSet(lfm.ratingDf.Series[0])
	qDistinctEle := distinctSeriesConvertSet(lfm.ratingDf.Series[1])
	pClasslen := lfm.classCount * pDistinctEle.Size()
	qClasslen := lfm.classCount * qDistinctEle.Size()
	pData := GenerateUniformArr(uni, int64(pClasslen))
	qData := GenerateUniformArr(uni, int64(qClasslen))
	lfm.itemidSet = &qDistinctEle
	lfm.useridSet = &pDistinctEle
	lfm.modelPfactor = mat.NewDense(pDistinctEle.Size(), lfm.classCount, pData)
	lfm.modelQfactor = mat.NewDense(qDistinctEle.Size(), lfm.classCount, qData)
	lfm.generateIdsMatrixIndexMapping()
	lfm.generateUserIdItemIdIndexDict()

}

func (lfm *LFM) generateIdsMatrixIndexMapping() {
	var userIdArr = make([]float64, 0)
	var itemIdArr = make([]float64, 0)
	lfm.useridSet.Iterator(func(v interface{}) bool {
		userIdArr = append(userIdArr, v.(float64))
		return true
	})
	lfm.itemidSet.Iterator(func(v interface{}) bool {
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
	lfm.idsMapping = idMapping
}

func distinctSeriesConvertSet(series dataframe.Series) gset.Set {
	seriesArr := series.(*dataframe.SeriesFloat64).Values
	var newSet = gset.New(true)
	for _, ele := range seriesArr {
		newSet.Add(ele)
	}
	return *newSet
}

func (lfm *LFM) _preidct(userId, itemId float64) (float64, float64) {
	p := lfm.modelPfactor.RowView(int(userId)).(*mat.VecDense)     //5,1
	q := lfm.modelQfactor.RowView(int(itemId)).(*mat.VecDense).T() // 1.5
	var res mat.Dense
	res.Mul(p, q) //5,5
	resRow, resCol := res.Caps()
	matR := mat.Sum(&res)
	logit := 1.0 / (1.0 + math.Exp(-matR))
	fmt.Println("predict res shape  matR  logit ", resRow, resCol, matR, logit)
	return logit, matR
}

func (lfm *LFM) _loss(userIndex, itemIndex, rating float64, step int) float64 {
	logit, matR := lfm._preidct(userIndex, itemIndex)
	costError := rating - matR
	fmt.Println("step : {} ,cost error : {}, logit  y  ", step, costError, logit, rating)
	return costError
}

func (lfm *LFM) _optimize(userIndex, itemIndex, e float64) {
	modp_r, modp_c := lfm.modelPfactor.Caps()
	fmt.Println("modelPfactor shape  ", modp_r, modp_c)
	modq_r, modq_c := lfm.modelQfactor.Caps()
	fmt.Println("modelqfactor sh ape  ", modq_r, modq_c)
	pVal := lfm.modelPfactor.RowView(int(userIndex)).(*mat.VecDense)
	qVal := lfm.modelQfactor.RowView(int(itemIndex)).(*mat.VecDense)
	var gradient_p, l2_p, grad_l2_p, delta_p, gradient_q, l2_q, grad_l2_q, delta_q, np_val, nq_val mat.VecDense
	gradient_p.ScaleVec(-e, qVal)
	g_r, g_c := gradient_p.Caps()
	p_r, p_c := pVal.Caps()
	q_r, q_c := qVal.Caps()
	fmt.Println("gradient_p shape  ", g_r, g_c)
	fmt.Println("pval shape ", p_r, p_c)
	fmt.Println("qval shape ", q_r, q_c)
	l2_p.ScaleVec(lfm.lam, pVal)
	l2_p_r, l2_p_c := l2_p.Caps() //shape 5,1
	fmt.Println("l2_p shape ", l2_p_r, l2_p_c)
	grad_l2_p.AddVec(&gradient_p, &l2_p)
	delta_p.ScaleVec(lfm.lr, &grad_l2_p)
	gradient_q.ScaleVec(-e, pVal)
	l2_q.ScaleVec(lfm.lam, qVal)
	grad_l2_q.AddVec(&gradient_q, &l2_q)
	delta_q.ScaleVec(lfm.lr, &grad_l2_q)
	np_val.SubVec(pVal, &delta_p)
	nq_val.SubVec(qVal, &delta_q)
	lfm.modelPfactor.SetRow(int(userIndex), vectofloat(&np_val))
	lfm.modelQfactor.SetRow(int(itemIndex), vectofloat(&nq_val))
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

func (lfm *LFM) Train() {
	fmt.Println("lfm training ", lfm.iterCount)
	for step := 0; step < lfm.iterCount; step++ {
		fmt.Println(" lfm step ", step)
		for userIdIndex, valDict := range lfm.userIndexItemIndexDict {
			itemIdIndexs := reflect.ValueOf(valDict).MapKeys()
			fmt.Println("itemids len : ,userid ", len(itemIdIndexs), userIdIndex)
			for itemId := range itemIdIndexs {
				costErr := lfm._loss(userIdIndex, float64(itemId), valDict[float64(itemId)], step)
				lfm._optimize(userIdIndex, float64(itemId), costErr)
				fmt.Println("step  costerror ", step, costErr)
			}
		}
		lfm.lr *= 0.9
	}
	fmt.Println("model training complete !!!!")
}

func (lfm *LFM) fromSeriesGenerateGSet(fieldName string) *gset.Set {
	itemIndex, _ := lfm.ratingDf.NameToColumn(fieldName)
	itemIds := lfm.ratingDf.Series[itemIndex].(*dataframe.SeriesFloat64).Values
	itemIdsInterfaceArr := floatArrConvertInterfaceArr(itemIds)
	itemIdsSet := gset.New(true)
	itemIdsSet.Add(itemIdsInterfaceArr...)
	return itemIdsSet
}

func (lfm *LFM) Predict(userId float64, topN int32) PairList {
	fmt.Println("now predict userid ", userId)
	labelSet := gset.New(true)
	labelArr := dfWhere(lfm.ratingDf, lfm.featureName, lfm.labelName, userId)
	labelInterfaceArr := floatArrConvertInterfaceArr(labelArr)
	labelSet.Add(labelInterfaceArr...)
	otherItemIds := lfm.itemidSet.Diff(labelSet)
	fmt.Println("otherItemIds  size :  label set size  ", otherItemIds.Size(), labelSet.Size())
	interestScoreDict := make(map[float64]float64, 0)
	userIndex := lfm.idsMapping.userIdIndexDict[userId]
	for _, itemId := range otherItemIds.Slice() {
		itemIndex := lfm.idsMapping.itemIdIndexDict[itemId.(float64)]
		_, score := lfm._preidct(userIndex, itemIndex)
		interestScoreDict[itemId.(float64)] = score
	}
	rankList := RankSortDict(interestScoreDict)
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

func RankSortDict(dict map[float64]float64) PairList {
	pl := make(PairList, len(dict))
	i := 0
	for k, v := range dict {
		pl[i] = Pair{k, v}
		i++
	}
	sort.Sort(sort.Reverse(pl))
	return pl
}

func floatArrConvertInterfaceArr(labelArr []float64) []interface{} {
	labelInterfaceArr := make([]interface{}, len(labelArr))
	for i, v := range labelArr {
		labelInterfaceArr[i] = v
	}
	return labelInterfaceArr
}

func dfWhere(df *dataframe.DataFrame, feature, label string, featureVal float64) []float64 {
	lableIndex, _ := df.NameToColumn(label)
	featureIndex, _ := df.NameToColumn(feature)
	labelArr := make([]float64, 0)
	labelSeries := df.Series[lableIndex].(*dataframe.SeriesFloat64)
	featureSeries := df.Series[featureIndex].(*dataframe.SeriesFloat64)
	for index, valz := range featureSeries.Values {
		if valz == featureVal {
			labelVal := labelSeries.Values[index]
			labelArr = append(labelArr, labelVal)
		}
	}
	return labelArr
}

func (lfm *LFM) dataFrameConvertDenseMatrix(df *dataframe.DataFrame) *mat.Dense {
	dataArr := seriesConvertFloatarray(df)
	matrixRow := df.NRows()
	matrixCol := len(df.Names())
	userItemRatingMatrix := mat.NewDense(matrixRow, matrixCol, dataArr)
	return userItemRatingMatrix
}

func seriesConvertFloatarray(df *dataframe.DataFrame) []float64 {
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

//func (lfm *LFM)evaluateLFMModel(selectCount int )float64{
//	rmae := float64(0)
//
//	return rmae
//}
//
//func (LFM *LFM) saveModel(savePath string){
//
//}
//
//func loadModel(loadPath string) *LFM{
//
//}
//
//func (lfm *LFM)similarItemRecommend(itemId float64)PairList{
//
//}
//
//func (lfm *LFM)rankSelectItemRecommend(item float64,selectItemSet gset.Set)PairList{
//
//}
