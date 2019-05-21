'''
Code to get best models
'''
import pandas as pd

def get_best_models(df, test_years, cols, metric):
    best_models = pd.DataFrame(columns= cols)

    for year in test_years:
        year_data = df[df['year']==year]
        highest = year_data[metric].max()
        model = year_data[year_data[metric] == highest]
        print("For year {}, highest {} attained is {}".format(year, metric, highest))
        best_models = best_models.append(model[cols])

    return best_models

