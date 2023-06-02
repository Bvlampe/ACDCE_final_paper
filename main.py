import dataprep
import models
import dataexpl


def main():
    dataprep.dataprep("merge")
    dataexpl.dataexpl()
    models.models()


if __name__ == '__main__':
    main()
