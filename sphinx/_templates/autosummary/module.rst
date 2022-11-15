{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :members:

{% block modules %}
{% if modules %}

**Sub-Modules**

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
