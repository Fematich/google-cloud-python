# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Client for interacting with the Google Cloud ML API."""


from google.cloud.client import JSONClient
from google.cloud.exceptions import make_exception
from google.cloud.ml.connection import Connection
from google.cloud.ml.project import Project


class Client(JSONClient):
    """Client to bundle configuration needed for API requests.

    :type project: str
    :param project: the project which the client acts on behalf of.
                    If not passed, falls back to the default inferred
                    from the environment.

    :type credentials: :class:`oauth2client.client.OAuth2Credentials` or
                       :class:`NoneType`
    :param credentials: The OAuth2 Credentials to use for the connection
                        owned by this client. If not passed (and if no ``http``
                        object is passed), falls back to the default inferred
                        from the environment.

    :type http: :class:`httplib2.Http` or class that defines ``request()``.
    :param http: An optional HTTP object to make requests. If not passed, an
                 ``http`` object is created that is bound to the
                 ``credentials`` for the current object.
    """

    _connection_class = Connection

    def get_config(self):
        """Get the service account information associated with your project. 
        You need this information in order to grant the service account
        persmissions for the Google Cloud Storage location where you put your
        model training code for training the model with Google Cloud Machine
        Learning.

        :rtype: dict
        :returns: Dictionary containing ServiceAccount and 
        ServiceAccountProject
        """

        project_name = 'projects/' + self.project
        response = self.connection.api_request(method='GET',
                                               path=project_name +
                                               ':getConfig')
        return response

    def predict(self, modelname, instances):
        """Performs prediction on the instances data in the request.

        :type modelname: str
        :param modelname: Name of the model that needs to perform the
        predictions

        :type instances: list
        :param instances: List of dictionaries containing the input data
        for model predictions.

        :rtype: list
        :returns: List of predictions represented as dictionaries, the keys
        and values of the dictionaries are defined by the model.
        :raises: ValueError if the model prediction returns an error.
        """
        data = {'instances': instances}
        model_name = 'projects/' + self.project + '/models/' + modelname
        response = self.connection.api_request(method='POST',
                                               path=modelname +
                                               ':predict',
                                               data=data)

        if "error" in response:
            raise make_exception(response, response["error"])
        else:
            return response['predictions']

