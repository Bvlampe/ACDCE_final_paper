import dataprep
import models
import dataexpl


def main():
    # dataprep.dataprep("merge")
    # dataexpl.dataexpl()
    models.models(cut_before_1990=True)


if __name__ == '__main__':
    main()
