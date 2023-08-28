from enum import Enum


class ManipulateMode(str, Enum):
    """
    how to train the classifier to manipulate
    """
    # train on whole celeba attr dataset
    celebahq_all = 'celebahq_all'
    # celeba with D2C's crop
    d2c_fewshot = 'd2cfewshot'
    d2c_fewshot_allneg = 'd2cfewshotallneg'
    # glioma public dataset
    gliomapublic = 'gliomapublic'

    def is_celeba_attr(self):
        return self in [
            ManipulateMode.d2c_fewshot,
            ManipulateMode.d2c_fewshot_allneg,
            ManipulateMode.celebahq_all,
        ]

    def is_single_class(self):
        return self in [
            ManipulateMode.d2c_fewshot,
            ManipulateMode.d2c_fewshot_allneg,
        ]

    def is_fewshot(self):
        return self in [
            ManipulateMode.d2c_fewshot,
            ManipulateMode.d2c_fewshot_allneg,
        ]

    def is_fewshot_allneg(self):
        return self in [
            ManipulateMode.d2c_fewshot_allneg,
        ]
