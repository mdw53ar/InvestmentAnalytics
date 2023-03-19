from datetime import datetime


class StringValidation:
    """
    class validating of the value passed is a string
    """

    def __set_name__(self, owner, name):
        self.name = name

    def __set__(self, instance, value):

        if not isinstance(value, str):
            raise AttributeError(f"Should be a string. Value passed: {value}")

        instance.__dict__[self.name] = value

    def __get__(self, instance, owner):
        if instance is None:
            return self

        return instance.__dict__[self.name]


class LetterValidation(StringValidation):
    """
    class validating if the value passed is a string and
    only contains letter. Inherits from StringValidation and adds one criteria to the setter/validation
    """

    def __set__(self, instance, value):
        value_strip = str(value).replace(" ", "")

        if not isinstance(value, str):
            raise AttributeError(f"Should be a string. Value passed: {value}")
        if not value_strip.isalpha() :
            raise AttributeError(f"Should only contain letters. Value passed: {value}")
        instance.__dict__[self.name] = value


class NumericValidation:
    """
    class validating if the value passed is numeric
    """

    def __set_name__(self, owner, name):
        self.name = name

    def __set__(self, instance, value):
        if not ((isinstance(value, int)) or (isinstance(value, float))):
            raise AttributeError('Must be numeric!')
        instance.__dict__[self.name] = value

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]


class DateValidation:
    """
    class validating if the value passed is a string with in
    yyyy-mm-dd format
    """

    def __set_name__(self, owner, name):
        self.name = name

    def __set__(self, instance, value):
        try:
            datetime.strptime(value, "%Y-%m-%d")
            instance.__dict__[self.name] = value
        except ValueError:
            print("Must be in yyyy-mm-dd string format!")

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]
