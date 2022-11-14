import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing   import StandardScaler
import tensorflow as tf
import random as rd
from predictor.AbstractPredictor import AbstractPredictor

class RicoPredict(AbstractPredictor):

    def predict(self):
        # datasetRanking = pd.read_csv('/home/ricardors/projetos/pauloctopus/paul-octopus-python/predictor/files/ranking.csv')
        # historicalResults = pd.read_csv('/home/ricardors/projetos/pauloctopus/paul-octopus-python/predictor/files/historical-results.csv')
        # matchSchedules = pd.read_csv('/home/ricardors/projetos/pauloctopus/paul-octopus-python/predictor/files/matches-schedule.csv')

        datasetRanking = pd.read_csv('ranking.csv')
        historicalResults = pd.read_csv('historical-results.csv')
        matchSchedules = pd.read_csv('matches-schedule.csv')


        ##lista times da copa...

        list_countries = set()
        for row in matchSchedules.iterrows():
            list_countries.add(row["country1"])
            list_countries.add(row["country2"])

        _year = "2022-10-06"
        #teamRanking = datasetRanking[(datasetRanking["country_full"] == _teamMain) & (datasetRanking["rank_date"] == _year)]

        #list_countries.add("Morocco")

        resultChallengers = []
        for teamCountry in list_countries:
            _teamMain = teamCountry

            print("###== TEAM MAIN "+ _teamMain+" ==####")

            _matches = []
            _indexRow = 0

            def transform_contry(_name):
                if _name == "USA":
                    return "United States"
                else:
                    return _name

            def calculeMatchesHome(historicVersus, _indexRow):
                for index, row in historicVersus.iterrows():    
                    yearOfMatch = row["date"][:4]
                    _indexRow += 1
                    
                    startYear = (yearOfMatch+"-01-01")
                    endYear = (yearOfMatch+"-12-31")

                    teamOthersRanking = datasetRanking[(datasetRanking["country_full"] == row["away_team"])]
                    teamOthersRanking = teamOthersRanking[(teamOthersRanking["rank_date"] >= startYear) & (teamOthersRanking["rank_date"] <= endYear)]

                    lengthOtherRanking = len(teamOthersRanking)
                    rating = 0

                    _derrota=0
                    _vitoria=0
                    if row["home_score"] > row["away_score"]:
                        _vitoria=1
                    elif row["home_score"] < row["away_score"]:
                        _derrota=-1
            
                    if lengthOtherRanking > 0:
                        rating = teamOthersRanking.iloc[0]["total_points"]
                        
                    _matches.append({
                        "team":row["away_team"],
                        "date": row["date"],
                        "rating":rating,
                        "result": _vitoria+_derrota,
                        "vitoria": _vitoria,
                        "derrota": _derrota,
                        "goalpro":row["home_score"],
                        "goalcontra":row["away_score"] 
                    })

            def calculeMatchesAway(historicVersus, _indexRow):
                for index, row in historicVersus.iterrows():    
                    yearOfMatch = row["date"][:4]
                    _indexRow += 1
                    
                    startYear = (yearOfMatch+"-01-01")
                    endYear = (yearOfMatch+"-12-31")

                    teamOthersRanking = datasetRanking[(datasetRanking["country_full"] == row["home_team"])]
                    teamOthersRanking = teamOthersRanking[(teamOthersRanking["rank_date"] >= startYear) & (teamOthersRanking["rank_date"] <= endYear)]

                    lengthOtherRanking = len(teamOthersRanking)
                    rating = 0

                    if lengthOtherRanking > 0:
                        rating = teamOthersRanking.iloc[0]["total_points"]


                    _derrota=0
                    _vitoria=0
                    if row["home_score"] > row["away_score"]:
                        _derrota=-1
                    elif row["home_score"] < row["away_score"]:
                        _vitoria=1
                    
                    _matches.append({
                        "team":row["home_team"],
                        "date": row["date"],
                        "rating":rating,
                        "result": _vitoria+_derrota,
                        "vitoria": _vitoria,
                        "derrota": _derrota,
                        "goalpro": row["away_score"],
                        "goalcontra":row["home_score"] 
                    })
                    
            historicTeam = historicalResults[(historicalResults["home_team"]==transform_contry(_teamMain)) & (historicalResults["tournament"]=="FIFA World Cup") & (historicalResults["date"] >= "1960-01-01")]
            calculeMatchesHome(historicTeam, _indexRow)

            historicTeam = historicalResults[(historicalResults["away_team"]==transform_contry(_teamMain)) & (historicalResults["tournament"]=="FIFA World Cup") & (historicalResults["date"] >= "1960-01-01")]
            calculeMatchesAway(historicTeam, _indexRow)

            if len(_matches) == 0:
                historicTeam = historicalResults[(historicalResults["home_team"]==transform_contry(_teamMain)) & (historicalResults["tournament"]!="FIFA World Cup")]
                calculeMatchesHome(historicTeam, _indexRow)

                historicTeam = historicalResults[(historicalResults["away_team"]==transform_contry(_teamMain)) & (historicalResults["tournament"]!="FIFA World Cup")]
                calculeMatchesAway(historicTeam, _indexRow)



            # dataFrameMatch.plot(x='rating', y='result', style='o')  
            # plt.title('rating vs result')  
            # plt.xlabel('rating')  
            # plt.ylabel('result')  
            # plt.show()


            #----------------------

            

            # x = dataFrameMatch.iloc[:,3].values.reshape(-1, 1)
            # y = dataFrameMatch.iloc[:,4].values.reshape(-1, 1)


            # plt.plot(x,y,"o")
            # plt.show()


            #Realizando Predições
            nextMatchesSchedule = matchSchedules[(matchSchedules["country1"]==_teamMain)]
            challengers = []

            for index, row in nextMatchesSchedule.iterrows():  
                challenger = row["country1"]
                if row["country1"] == _teamMain: #logica praa obter o adversario
                    challenger = row["country2"]
                    
                teamOthersRanking = datasetRanking[(datasetRanking["country_full"] == challenger)]
                teamOthersRanking = teamOthersRanking[(teamOthersRanking["rank_date"] == _year)]

                lengthOtherRanking = len(teamOthersRanking)
                rating = 0

                if lengthOtherRanking > 0:
                    rating = teamOthersRanking.iloc[0]["total_points"]

                challengers.append({
                    "challenger": challenger,
                    "country1": row["country1"],
                    "country2": row["country2"],
                    "rating": rating,
                    "result": 0
                })

            
            dataFrameMatch = pd.DataFrame(_matches)
            x = dataFrameMatch['rating'].values.reshape(-1, 1)
            y = dataFrameMatch['result'].values.reshape(-1, 1)

            dfchallengers = pd.DataFrame(challengers)

            firstMatch = []
            secondMatch = []
            thirdMatch = []
            #1,2,3,4,5,6,7,8,9,10
            for xtime in [1,2,3,4,5]:
                # Treino o Conjunto de Testes
                #x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,shuffle = True)


                #modelo de regressão
                colum_x = [tf.feature_column.numeric_column("x",shape=[1])]
                regressor = tf.estimator.LinearRegressor(feature_columns=colum_x)

                #Treinando o modelo 
                # conjunto_treinamento = tf.compat.v1.estimator.inputs.numpy_input_fn({"x":x_train},y_train,batch_size = 32,
                #                                                                 num_epochs = 1000,shuffle = True)
                conjunto_treinamento = tf.compat.v1.estimator.inputs.numpy_input_fn({"x":x},y,batch_size = 32,
                                                                                num_epochs = 1000,shuffle = True)


                regressor.train(conjunto_treinamento,steps = 10000)
                metricas_treino = regressor.evaluate(conjunto_treinamento,steps =10000)

                # #Testando o Modelo
                # conjunto_teste = tf.compat.v1.estimator.inputs.numpy_input_fn({"x":x_test},y_test,batch_size = 32,
                #                                                                 num_epochs = 1000,shuffle = False)
                                                                                
                # #metricas_teste
                # metricas_teste = regressor.evaluate(conjunto_teste,steps = 1000)

                # print("METRICAS TREINO")
                # print(metricas_treino)
                # print("METRICAS TESTE")
                # print(metricas_teste)
                # print("===")
                # if metricas_treino["loss"] > metricas_teste["loss"]:
                #     print("O Conjunto de Teste teve um desempenho melhor que o conjunto de treino, com um erro de:", metricas_teste["loss"])
                # else:
                #     print("O Conjunto de Treino teve um desempenho melhor que o conjunto de teste, com um erro de:", metricas_treino["loss"])

                predict_x = dfchallengers['rating'].values.reshape(-1, 1)
                predict_model = tf.compat.v1.estimator.inputs.numpy_input_fn({"x":predict_x}, batch_size=32, shuffle = False)
                for idx, prediction in enumerate(regressor.predict(input_fn = predict_model)):
                    if idx == 0:
                        firstMatch.append(prediction["predictions"][0])
                    elif idx == 1:
                        secondMatch.append(prediction["predictions"][0])
                    else:
                        thirdMatch.append(prediction["predictions"][0])

            #-- removendo outliers...
            if len(firstMatch) > 0:
                firstMatch.sort(reverse=True)
                firstMatch.pop()
                firstMatch.sort(reverse=False)
                firstMatch.pop()
                challengers[0]["result"] = round(sum(firstMatch)/len(firstMatch), 2)

            if len(secondMatch) > 0:
                secondMatch.sort(reverse=True)
                secondMatch.pop()
                secondMatch.sort(reverse=False)
                secondMatch.pop()
                challengers[1]["result"] = round(sum(secondMatch)/len(secondMatch), 2)


            if len(thirdMatch) > 0:
                thirdMatch.sort(reverse=True)
                thirdMatch.pop()
                thirdMatch.sort(reverse=False)
                thirdMatch.pop()
                challengers[2]["result"] = round(sum(thirdMatch)/len(thirdMatch), 2)

            print(challengers)

            #---Relacionar as vitórias com os goals... challanger vs Goals..????

            ##------ treinar goals... removendo jogos com mais de 4 goals...
            goalsVitoria = list(filter(lambda match: match["vitoria"] == 1 and match["goalpro"] <= 3, _matches))
            goalsEmpate = list(filter(lambda match: match["result"] == 1 and match["goalpro"] <= 4, _matches))
            goalsEDerrota = list(filter(lambda match: match["derrota"] == -1 and match["goalcontra"] <= 4, _matches))

            #modelo de regressão
            dfgoalsVitoria = pd.DataFrame(goalsVitoria)

            #- > 0.20 = vitória
            #- < 0.5 = derrota
            #- 0.5 ~ 0.20 = empate
            indice_vitoria = 0.15
            indice_derrota = -0.4

            ### VITORIAS
            willWinChallengers = list(filter(lambda match: match["result"] > indice_vitoria, challengers))
            for challenger in willWinChallengers:
                #--Goals Pro
                x = dfgoalsVitoria['rating'].values.reshape(-1, 1)
                y = dfgoalsVitoria['goalpro'].values.reshape(-1, 1)

                colum_x = [tf.feature_column.numeric_column("x",shape=[1])]
                regressorGoalsProd = tf.estimator.LinearRegressor(feature_columns=colum_x)

                conjunto_treinamento = tf.compat.v1.estimator.inputs.numpy_input_fn({"x":x},y,batch_size = 32,
                                                                                    num_epochs = 1000, shuffle = True)
                regressorGoalsProd.train(conjunto_treinamento,steps = 10000)
                metricas_treino = regressorGoalsProd.evaluate(conjunto_treinamento,steps =10000)

                dfchallengers = pd.DataFrame([challenger])
                predict_x = dfchallengers['rating'].values.reshape(-1, 1)
                predict_model = tf.compat.v1.estimator.inputs.numpy_input_fn({"x":predict_x}, batch_size=32, shuffle = False)
                goalsPro = []
                for idx, prediction in enumerate(regressorGoalsProd.predict(input_fn = predict_model)):
                    goalsPro.append({"goal": rd.randint(1, max(1, int(round(float(prediction["predictions"]), 0))))})


                print(goalsPro)

                #--Goals Contra
                x = dfgoalsVitoria['goalpro'].values.reshape(-1, 1)
                y = dfgoalsVitoria['goalcontra'].values.reshape(-1, 1)

                colum_x = [tf.feature_column.numeric_column("x",shape=[1])]
                regressorGoalsContra = tf.estimator.LinearRegressor(feature_columns=colum_x)

                conjunto_treinamento = tf.compat.v1.estimator.inputs.numpy_input_fn({"x":x},y,batch_size = 32,
                                                                                    num_epochs = 1000,shuffle = True)
                regressorGoalsContra.train(conjunto_treinamento,steps = 10000)
                metricas_treino = regressorGoalsContra.evaluate(conjunto_treinamento,steps =10000)

                dfgoalsPro = pd.DataFrame(goalsPro)
                predict_x = dfgoalsPro["goal"].values.reshape(-1, 1)
                predict_model = tf.compat.v1.estimator.inputs.numpy_input_fn({"x":predict_x}, batch_size=32, shuffle = False)
                goalsContra = []
                for idx, prediction in enumerate(regressorGoalsContra.predict(input_fn = predict_model)):
                    goalsContra.append(rd.randint(0, int(round(float(prediction["predictions"]), 0))))
                    
                for idx, goal in enumerate(goalsPro):
                    if _teamMain == challenger["country1"]:
                        resultChallengers.append({
                            "home": challenger["country1"],
                            "home_score": str(goal["goal"]),
                            "away_score": str(goalsContra[idx]),
                            "away": challenger["country2"]
                        })
                    else :
                        resultChallengers.append({
                            "home": challenger["country1"],
                            "home_score": str(goalsContra[idx]),
                            "away_score": str(goal["goal"]),
                            "away": challenger["country2"]
                        })
            
            dfgoalsDerrota = pd.DataFrame(goalsEDerrota)
            willWinChallengers = list(filter(lambda match: match["result"] <= indice_derrota, challengers))
            for challenger in willWinChallengers:
                #--Goals Contra
                x = dfgoalsDerrota['rating'].values.reshape(-1, 1)
                y = dfgoalsDerrota['goalcontra'].values.reshape(-1, 1)

                colum_x = [tf.feature_column.numeric_column("x",shape=[1])]
                regressorGoalsContra = tf.estimator.LinearRegressor(feature_columns=colum_x)

                conjunto_treinamento = tf.compat.v1.estimator.inputs.numpy_input_fn({"x":x},y,batch_size = 32,
                                                                                    num_epochs = 1000, shuffle = True)
                regressorGoalsContra.train(conjunto_treinamento,steps = 10000)
                metricas_treino = regressorGoalsContra.evaluate(conjunto_treinamento,steps =10000)

                dfchallengers = pd.DataFrame([challenger])
                predict_x = dfchallengers['rating'].values.reshape(-1, 1)
                predict_model = tf.compat.v1.estimator.inputs.numpy_input_fn({"x":predict_x}, batch_size=32, shuffle = False)
                goalsContra = []
                for idx, prediction in enumerate(regressorGoalsContra.predict(input_fn = predict_model)):
                    goalsContra.append({"goal": rd.randint(1, max(1, int(round(float(prediction["predictions"]), 0))))})

                print(goalsContra)

                #--Goals Pró
                x = dfgoalsDerrota['goalcontra'].values.reshape(-1, 1)
                y = dfgoalsDerrota['goalpro'].values.reshape(-1, 1)

                colum_x = [tf.feature_column.numeric_column("x",shape=[1])]
                regressorGoalsPro = tf.estimator.LinearRegressor(feature_columns=colum_x)

                conjunto_treinamento = tf.compat.v1.estimator.inputs.numpy_input_fn({"x":x},y,batch_size = 32,
                                                                                    num_epochs = 1000,shuffle = True)
                regressorGoalsPro.train(conjunto_treinamento,steps = 10000)
                metricas_treino = regressorGoalsPro.evaluate(conjunto_treinamento,steps =10000)

                dfgoalsPro = pd.DataFrame(goalsContra)
                predict_x = dfgoalsPro["goal"].values.reshape(-1, 1)
                predict_model = tf.compat.v1.estimator.inputs.numpy_input_fn({"x":predict_x}, batch_size=32, shuffle = False)
                goalsPro = []
                for idx, prediction in enumerate(regressorGoalsPro.predict(input_fn = predict_model)):
                    goalsPro.append(rd.randint(0, int(round(float(prediction["predictions"]), 0))))
                    
                for idx, goal in enumerate(goalsContra):
                    goalpro = goalsPro[idx]
                    goalcontra = goal["goal"]
                    if goalcontra == goalpro:
                        goalpro = goalpro-1

                    if _teamMain == challenger["country1"]:
                        resultChallengers.append({
                            "home": challenger["country1"],
                            "home_score": goalpro,
                            "away_score": str(goalcontra),
                            "away": challenger["country2"]
                        })
                    else :
                        resultChallengers.append({
                            "home": challenger["country1"],
                            "home_score": str(goalcontra),
                            "away_score": goalpro,
                            "away": challenger["country2"]
                        })


            ### EMPATE
            dfgoalsEmpate = pd.DataFrame(goalsEmpate)
            willWinChallengers = list(filter(lambda match: match["result"] > indice_derrota and match["result"] <= indice_vitoria, challengers))
            for challenger in willWinChallengers:
                #--Goals Contra
                x = dfgoalsEmpate['rating'].values.reshape(-1, 1)
                y = dfgoalsEmpate['goalcontra'].values.reshape(-1, 1)

                colum_x = [tf.feature_column.numeric_column("x",shape=[1])]
                regressorGoalsEmpate = tf.estimator.LinearRegressor(feature_columns=colum_x)

                conjunto_treinamento = tf.compat.v1.estimator.inputs.numpy_input_fn({"x":x},y,batch_size = 32,
                                                                                    num_epochs = 1000, shuffle = True)
                regressorGoalsEmpate.train(conjunto_treinamento,steps = 10000)
                metricas_treino = regressorGoalsEmpate.evaluate(conjunto_treinamento,steps =10000)

                dfchallengers = pd.DataFrame([challenger])
                predict_x = dfchallengers['rating'].values.reshape(-1, 1)
                predict_model = tf.compat.v1.estimator.inputs.numpy_input_fn({"x":predict_x}, batch_size=32, shuffle = False)
                for idx, prediction in enumerate(regressorGoalsEmpate.predict(input_fn = predict_model)):
                    goal = max(0, int(round(float(prediction["predictions"]), 0)))
                    resultChallengers.append({
                            "home": challenger["country1"],
                            "home_score": goal,
                            "away_score": goal,
                            "away": challenger["country2"]})
        
        return resultChallengers

