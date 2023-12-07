import sastvd.helpers.datasets as svdd
import sastvd.ivdetect.evaluate as ivde


def bigvul():
    """Run preperation scripts for BigVul dataset."""
    svdd.bigvul()
    ivde.get_dep_add_lines_bigvul()
    print("Glove Model")
    svdd.generate_glove("bigvul")
    print("Big Vul")
    svdd.generate_d2v("bigvul")


if __name__ == "__main__":
    bigvul()
