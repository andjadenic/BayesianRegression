import pandas as pd
import numpy as np


def convert_to_hours(input_list: list):
        """
        Method takes list of times in format HH:MM:SS
        and returns corresponding numpy array of float number of hours
        """
        output_list = []
        for time_str in input_list:
                if time_str != '--:--:--':
                        hours, minutes, seconds = map(float, time_str.split(':'))
                        total_hours = hours + (minutes / 60) + (seconds / 3600)
                        output_list.append(total_hours)
        output_list = np.array(output_list)
        return output_list


if __name__ == "__main__":
        # Setting the option to show all columns from DataFrame
        pd.options.display.max_columns = None

        # Read results from 2018. and 2019.
        df_2018 = pd.read_csv(r'ironman_data\2018IRONMANWorldChampionship.csv',
                              delimiter=';')
        df_2019 = pd.read_csv(r'ironman_data\2019IRONMANWorldChampionship.csv', delimiter=';')

        # Take a look at the data
        print(df_2018.head())
        print(df_2019.head())
        print(df_2018.tail())
        print(df_2019.tail())

        # Look only at athletes that finished triathlon
        df_2018 = df_2018[df_2018['Status'] == 'Finish']
        df_2019 = df_2019[df_2019['Status'] == 'Finish']

        # Make lists out of columns that represent run and overall times in format HH:MM:SS
        overall_list_2018 = df_2018['Overall'].tolist()
        run_list_2018 = df_2018['Run'].tolist()
        overall_list_2019 = df_2019['Overall'].tolist()
        run_list_2019 = df_2019['Run'].tolist()

        # Change the format to float number o hours in lists above
        overall_2018 = convert_to_hours(overall_list_2018)
        run_2018 = convert_to_hours(run_list_2018)
        overall_2019 = convert_to_hours(overall_list_2019)
        run_2019 = convert_to_hours(run_list_2019)

        # Export relevant data as a csv file
        df_output_2018 = pd.DataFrame({
                "run_2018": run_2018,
                "overall_2018": overall_2018
        })
        #df_output_2018.to_csv("ironman_data\data_2018_lin_reg.csv")
        df_output_2019 = pd.DataFrame({
                "run_2019": run_2019,
                "overall_2019": overall_2019
        })
        #df_output_2019.to_csv("ironman_data\data_2019_lin_reg.csv")

        # Visualise distribution of overall finish times in both years
        fig, ax = plt.subplots()
        ax.hist(overall_2018, bins=np.linspace(7, 17, 11), alpha=0.4, color='deeppink', label='2018. overall finish times')
        ax.hist(overall_2019, bins=np.linspace(7, 17, 11), alpha=0.4, color='orange', label='2019. overall finish times')
        ax.legend()
        ax.set_title('Histogram of 2018. and 2019. overall finish times (hrs)')
        ax.set_xlabel('2018. and 2019. overall finish times (hrs)')
        ax.set_ylabel('Frequency')
        #plt.show()

        # Visualise how overall finish times depends on run portion
        plt.figure(figsize=(10, 6))
        plt.scatter(run_2018, overall_2018, s=.5, alpha=.4, c='deeppink', label='2018. Ironman')
        plt.scatter(run_2019, overall_2019, s=.5, alpha=.4, c='red', label='2019. Ironman')
        plt.legend()
        #plt.show()
