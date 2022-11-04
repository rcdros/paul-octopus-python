import csv


def read_csv(file_name):
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        matches_data = [row for row in reader]

    return matches_data


def write_csv(data, csv_columns, file_name):
    with open(file_name, 'w', newline='') as fp:
        writer = csv.DictWriter(fp, delimiter=',', fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(data)
