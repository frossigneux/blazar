# Copyright (c) 2013 Bull.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.


policy_data = """
{

    "admin": "is_admin:True or role:admin or role:masterofuniverse",
    "admin_or_owner":  "rule:admin or project_id:%(project_id)s",
    "default": "!",

    "admin_api": "rule:admin",
    "climate:leases": "rule:admin_or_owner",
    "climate:oshosts": "rule:admin_api",

    "climate:leases:get": "rule:admin_or_owner",
    "climate:leases:create": "rule:admin_or_owner",
    "climate:leases:delete": "rule:admin_or_owner",
    "climate:leases:update": "rule:admin_or_owner",

    "climate:plugins:get": "@",

    "climate:oshosts:get": "rule:admin_api",
    "climate:oshosts:create": "rule:admin_api",
    "climate:oshosts:delete": "rule:admin_api",
    "climate:oshosts:update": "rule:admin_api"
}
"""
