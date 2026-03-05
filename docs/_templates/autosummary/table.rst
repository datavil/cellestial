CUSTOM TABLE
{% for item in items %}
* :obj:`{{ item.split('.')[-1] }} <{{ item }}>`
{%- endfor %}
