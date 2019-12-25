package main

import (
	"fmt"
	"github.com/alexflint/go-memdump"
	_ "github.com/alexflint/go-memdump"
	"lfm_rec/recCore"
	"os"
)

func main() {
	ratingPath := "./recCore/ratings.csv"
	var classCount, iterCount int = 5, 10
	var lr, lam float64 = 0.01, 0.02
	var featureName, labelName = "userId", "itemId"
	var featureNameArr = []string{"UserID", "MovieID", "Rating", "Timestamp"}
	lfm := recCore.NewDefaultLFM(ratingPath, classCount, iterCount, lr, lam, featureName, labelName, featureNameArr)
	lfm.InitModel()
	lfm.Train(5)
	lfm.EvaluateLFMModel(100, 10)
	userId := 23.0
	var topN int32 = 500
	var itemScore = lfm.Predict(userId, topN)
	userIndex := lfm.IdsMapping.UserIdIndexDict[userId]
	for index, itemScoreEle := range itemScore {
		itemId := float64(itemScoreEle.Key)
		fmt.Println("userindex , item : %f ,score : %f ", userIndex, itemId, itemScoreEle.Value)
		if index == int(topN) {
			break
		}
	}
	dumpPath := "./data.memdump"
	w, err := os.Create(dumpPath)
	if err != nil {
		fmt.Println("has error")
	}

	memdump.Encode(w, &lfm)
	r, err := os.Open("/tmp/data.memdump")
	if err != nil {
		fmt.Println("has error")
	}
	var lfmnew *recCore.LFM

	memdump.Decode(r, &lfmnew)
	var itemScorez = lfmnew.Predict(userId, topN)
	userIndexz := lfmnew.IdsMapping.UserIdIndexDict[userId]
	for index, itemScoreEle := range itemScorez {
		itemId := float64(itemScoreEle.Key)
		fmt.Println("userindex , item : %f ,score : %f ", userIndexz, itemId, itemScoreEle.Value)
		if index == int(topN) {
			break
		}
	}

}
