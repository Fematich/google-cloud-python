Using the API
=============

The `Google Cloud ML`_ API enables developers to convert audio to text.
The API recognizes over 80 languages and variants, to support your global user
base.

.. warning::

    This is a Beta release of Google Cloud ML API. This
    API is not intended for real-time usage in critical applications.

.. _Google Cloud ML: https://cloud.google.com/ml/docs

Client
------

:class:`~google.cloud.ml.client.Client` objects provide a
means to configure your application. Each instance holds
an authenticated connection to the Cloud ML Service.

For an overview of authentication in ``google-cloud-python``, see
:doc:`google-cloud-auth`.

Assuming your environment is set up as described in that document,
create an instance of :class:`~google.cloud.ml.client.Client`.

.. code-block:: python

    >>> from google.cloud import ml
    >>> client = ml.Client()
