{%- if obj.display %}
{{ obj.name }}
{{ "=" * obj.name|length }}

.. py:module:: {{ obj.name }}

{% if obj.docstring %}
{{ obj.docstring }}
{% endif %}

{% for member in obj.members %}
{% if member.type == "function" %}
{{ member.name }}
{{ "-" * member.name|length }}
{% endif %}
{{ member.render() }}
{% endfor %}

{% endif %}
