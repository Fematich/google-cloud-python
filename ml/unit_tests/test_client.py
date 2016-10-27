# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest


class TestClient(unittest.TestCase):

    def _getTargetClass(self):
        from google.cloud.ml.client import Client
        return Client

    def _makeOne(self, *args, **kw):
        return self._getTargetClass()(*args, **kw)

    def test_ctor(self):
        from google.cloud.ml.connection import Connection

        creds = _Credentials()
        http = object()
        client = self._makeOne(credentials=creds, http=http)
        self.assertIsInstance(client.connection, Connection)
        self.assertTrue(client.connection.credentials is creds)
        self.assertTrue(client.connection.http is http)

    def test_get_config(self):
        project = 'projectid'
        RETURNED = {'serviceAccount': 'ServiceAccount',
                    'serviceAccountProject': 'serviceAccountProject'}
        credentials = _Credentials()
        client = self._makeOne(credentials=credentials, project=project)
        client.connection = _Connection(RETURNED)

        response = client.get_config()

        self.assertEqual(len(client.connection._requested), 1)
        req = client.connection._requested[0]
        self.assertEqual(len(req), 2)
        self.assertEqual(req['method'], 'GET')
        self.assertEqual(req['path'], 'projects/' + project +
                         ':getConfig')

        expected = RETURNED
        self.assertEqual(response, expected)

    def test_predict(self):
        project = 'projectid'
        model = 'modelname'
        instances = [{'input_x': [0.1, 0.9]}, {'input_x': [0.75, 0.25]}]
        REQUEST = {'instances': instances}
        RETURNED = {'predictions': [{'label': 'beach', 'scores': [0.1, 0.9]},
                                    {'label': 'car', 'scores': [0.75, 0.25]}]}
        credentials = _Credentials()
        client = self._makeOne(credentials=credentials, project=project)
        client.connection = _Connection(RETURNED)

        response = client.predict(model, instances)

        self.assertEqual(len(client.connection._requested), 1)
        req = client.connection._requested[0]
        self.assertEqual(len(req), 3)
        self.assertEqual(req['data'], REQUEST)
        self.assertEqual(req['method'], 'POST')
        self.assertEqual(req['path'], 'projects/' + project + '/models' + model
                         ':predict')

        expected = RETURNED['predictions']
        self.assertEqual(response, expected)

    def test_predict(self):
        roject = 'projectid'
        model = 'modelname'
        instances = [{'input_x': [0.1, 0.9]}, {'input_x': [0.75, 0.25]}]
        REQUEST = {'instances': instances}
        RETURNED = {'error': 'error message'}

        credentials = _Credentials()
        client = self._makeOne(credentials=credentials, project=project)
        client.connection = _Connection(RETURNED)

        with self.assertRaises(ValueError):
            response = client.predict(model, instances)


class _Credentials(object):

    _scopes = ('https://www.googleapis.com/auth/cloud-platform',)

    def __init__(self, authorized=None):
        self._authorized = authorized
        self._create_scoped_calls = 0

    @staticmethod
    def create_scoped_required():
        return True

    def create_scoped(self, scope):
        self._scopes = scope
        return self


class _Connection(object):

    def __init__(self, *responses):
        self._responses = responses
        self._requested = []

    def api_request(self, **kw):
        self._requested.append(kw)
        response, self._responses = self._responses[0], self._responses[1:]
        return response
