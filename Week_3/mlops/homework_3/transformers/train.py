from typing import Tuple
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from mlops.utils.data_preparation.encoders import vectorize_features

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(data, *args, **kwargs) -> Tuple[DictVectorizer, LinearRegression  ]:
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    categorical = ['PULocationID', 'DOLocationID']
    target  = ['duration']

    y_train = data[target]
    X_train, _, dv = vectorize_features(training_set=data[categorical])
    
    # train the model
    model = LinearRegression()
    model.fit(X=X_train, y=y_train)
    print(f"The intercept of the fitted linear regression model is: {model.intercept_}")
    
    return dv, model
