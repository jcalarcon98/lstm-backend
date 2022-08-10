import pathlib
import pandas


class CountryManager:

    current_dir : str = pathlib.Path().resolve()

    @classmethod
    def get_countries(cls):
        train_file_path = f'{cls.current_dir}/app/data/train.csv'
        train_data = pandas.read_csv(train_file_path)
        countries = train_data['Country_Region'].unique()
        return list(countries)
