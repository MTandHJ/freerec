{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

   {% block classes %}
   {% if classes %}
   Classes
   -------

   .. autosummary::
      :toctree:
      :template: class.rst

   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   Functions
   ---------

   .. autosummary::
      :toctree:

   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
