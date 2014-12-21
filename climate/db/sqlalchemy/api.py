# Copyright (c) 2013 Mirantis Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of SQLAlchemy backend."""

import datetime
import json
import sys

import six

import sqlalchemy as sa
from sqlalchemy.sql.expression import asc
from sqlalchemy.sql.expression import desc

from climate.db import exceptions as db_exc
from climate.db import utils as db_utils
from climate.db.sqlalchemy import facade_wrapper
from climate.db.sqlalchemy import models
from climate.openstack.common.db import exception as common_db_exc
from climate.openstack.common.db import options as db_options
from climate.openstack.common.db.sqlalchemy import session as db_session
from climate.openstack.common.gettextutils import _
from climate.openstack.common import log as logging

from keystoneclient.v2_0 import client
import kwrankingclient.client as kwrclient

LOG = logging.getLogger(__name__)

get_engine = facade_wrapper.get_engine
get_session = facade_wrapper.get_session


def get_backend():
    """The backend is this module itself."""
    return sys.modules[__name__]


def model_query(model, session=None):
    """Query helper.

    :param model: base model to query
    :param project_only: if present and current context is user-type,
            then restrict query to match the project_id from current context.
    """
    session = session or get_session()

    return session.query(model)


def setup_db():
    try:
        engine = db_session.EngineFacade(db_options.CONF.database.connection,
                                         sqlite_fk=True).get_engine()
        models.Lease.metadata.create_all(engine)
    except sa.exc.OperationalError as e:
        LOG.error(_("Database registration exception: %s"), e)
        return False
    return True


def drop_db():
    try:
        engine = db_session.EngineFacade(db_options.CONF.database.connection,
                                         sqlite_fk=True).get_engine()
        models.Lease.metadata.drop_all(engine)
    except Exception as e:
        LOG.error(_("Database shutdown exception: %s"), e)
        return False
    return True


# Helpers for building constraints / equality checks


def constraint(**conditions):
    return Constraint(conditions)


def equal_any(*values):
    return EqualityCondition(values)


def not_equal(*values):
    return InequalityCondition(values)


class Constraint(object):
    def __init__(self, conditions):
        self.conditions = conditions

    def apply(self, model, query):
        for key, condition in self.conditions.iteritems():
            for clause in condition.clauses(getattr(model, key)):
                query = query.filter(clause)
        return query


class EqualityCondition(object):
    def __init__(self, values):
        self.values = values

    def clauses(self, field):
        return sa.or_([field == value for value in self.values])


class InequalityCondition(object):
    def __init__(self, values):
        self.values = values

    def clauses(self, field):
        return [field != value for value in self.values]


# Reservation
def _reservation_get(session, reservation_id):
    query = model_query(models.Reservation, session)
    return query.filter_by(id=reservation_id).first()


def reservation_get(reservation_id):
    return _reservation_get(get_session(), reservation_id)


def reservation_get_all():
    query = model_query(models.Reservation, get_session())
    return query.all()


def reservation_get_all_by_lease_id(lease_id):
    reservations = (model_query(models.Reservation,
                    get_session()).filter_by(lease_id=lease_id))
    return reservations.all()


def reservation_get_all_by_values(**kwargs):
    """Returns all entries filtered by col=value."""

    reservation_query = model_query(models.Reservation, get_session())
    for name, value in kwargs.items():
        column = getattr(models.Reservation, name, None)
        if column:
            reservation_query = reservation_query.filter(column == value)
    return reservation_query.all()


def reservation_create(values):
    values = values.copy()
    reservation = models.Reservation()
    reservation.update(values)

    session = get_session()
    with session.begin():
        try:
            reservation.save(session=session)
        except common_db_exc.DBDuplicateEntry as e:
            # raise exception about duplicated columns (e.columns)
            raise db_exc.ClimateDBDuplicateEntry(
                model=reservation.__class__.__name__, columns=e.columns)

    return reservation_get(reservation.id)


def reservation_update(reservation_id, values):
    session = get_session()

    with session.begin():
        reservation = _reservation_get(session, reservation_id)
        reservation.update(values)
        reservation.save(session=session)

    return reservation_get(reservation_id)


def reservation_destroy(reservation_id):
    session = get_session()
    with session.begin():
        reservation = _reservation_get(session, reservation_id)

        if not reservation:
            # raise not found error
            raise db_exc.ClimateDBNotFound(id=reservation_id,
                                           model='Reservation')

        session.delete(reservation)


# Lease
def _lease_get(session, lease_id):
    query = model_query(models.Lease, session)
    return query.filter_by(id=lease_id).first()


def lease_get(lease_id):
    return _lease_get(get_session(), lease_id)


def lease_get_all():
    query = model_query(models.Lease, get_session())
    return query.all()


def lease_get_all_by_project(project_id):
    raise NotImplementedError


def lease_get_all_by_user(user_id):
    raise NotImplementedError


def lease_list(project_id=None):
    query = model_query(models.Lease, get_session())
    if project_id is not None:
        query = query.filter_by(project_id=project_id)
    return query.all()


def lease_create(values):
    values = values.copy()
    lease = models.Lease()
    reservations = values.pop("reservations", [])
    events = values.pop("events", [])
    lease.update(values)

    session = get_session()
    with session.begin():
        try:
            lease.save(session=session)
        except common_db_exc.DBDuplicateEntry as e:
            # raise exception about duplicated columns (e.columns)
            raise db_exc.ClimateDBDuplicateEntry(
                model=lease.__class__.__name__, columns=e.columns)

        try:
            for r in reservations:
                reservation = models.Reservation()
                reservation.update({"lease_id": lease.id})
                reservation.update(r)
                reservation.save(session=session)
        except common_db_exc.DBDuplicateEntry as e:
            # raise exception about duplicated columns (e.columns)
            raise db_exc.ClimateDBDuplicateEntry(
                model=reservation.__class__.__name__, columns=e.columns)

        try:
            for e in events:
                event = models.Event()
                event.update({"lease_id": lease.id})
                event.update(e)
                event.save(session=session)
        except common_db_exc.DBDuplicateEntry as e:
            # raise exception about duplicated columns (e.columns)
            raise db_exc.ClimateDBDuplicateEntry(
                model=event.__class__.__name__, columns=e.columns)

    return lease_get(lease.id)


def lease_update(lease_id, values):
    session = get_session()

    with session.begin():
        lease = _lease_get(session, lease_id)
        lease.update(values)
        lease.save(session=session)

    return lease_get(lease_id)


def lease_destroy(lease_id):
    session = get_session()
    with session.begin():
        lease = _lease_get(session, lease_id)

        if not lease:
            # raise not found error
            raise db_exc.ClimateDBNotFound(id=lease_id, model='Lease')

        session.delete(lease)


# Event
def _event_get(session, event_id):
    query = model_query(models.Event, session)
    return query.filter_by(id=event_id).first()


def _event_get_all(session):
    query = model_query(models.Event, session)
    return query


def event_get(event_id):
    return _event_get(get_session(), event_id)


def event_get_all():
    return _event_get_all(get_session()).all()


def _event_get_sorted_by_filters(sort_key, sort_dir, filters):
    """Return an event query filtered and sorted by name of the field."""

    sort_fn = {'desc': desc, 'asc': asc}

    events_query = _event_get_all(get_session())

    if 'status' in filters:
        events_query = (
            events_query.filter(models.Event.status == filters['status']))
    if 'lease_id' in filters:
        events_query = (
            events_query.filter(models.Event.lease_id == filters['lease_id']))
    if 'event_type' in filters:
        events_query = events_query.filter(models.Event.event_type ==
                                           filters['event_type'])

    events_query = events_query.order_by(
        sort_fn[sort_dir](getattr(models.Event, sort_key))
    )

    return events_query


def event_get_first_sorted_by_filters(sort_key, sort_dir, filters):
    """Return first result for events

    Return the first result for all events matching the filters
    and sorted by name of the field.
    """

    return _event_get_sorted_by_filters(sort_key, sort_dir, filters).first()


def event_get_all_sorted_by_filters(sort_key, sort_dir, filters):
    """Return events filtered and sorted by name of the field."""

    return _event_get_sorted_by_filters(sort_key, sort_dir, filters).all()


def event_create(values):
    values = values.copy()
    event = models.Event()
    event.update(values)

    session = get_session()
    with session.begin():
        try:
            event.save(session=session)
        except common_db_exc.DBDuplicateEntry as e:
            # raise exception about duplicated columns (e.columns)
            raise db_exc.ClimateDBDuplicateEntry(
                model=event.__class__.__name__, columns=e.columns)

    return event_get(event.id)


def event_update(event_id, values):
    session = get_session()

    with session.begin():
        event = _event_get(session, event_id)
        event.update(values)
        event.save(session=session)

    return event_get(event_id)


def event_destroy(event_id):
    session = get_session()
    with session.begin():
        event = _event_get(session, event_id)

        if not event:
            # raise not found error
            raise db_exc.ClimateDBNotFound(id=event_id, model='Event')

        session.delete(event)


# ComputeHostReservation
def _host_reservation_get(session, host_reservation_id):
    query = model_query(models.ComputeHostReservation, session)
    return query.filter_by(id=host_reservation_id).first()


def host_reservation_get(host_reservation_id):
    return _host_reservation_get(get_session(),
                                 host_reservation_id)


def host_reservation_get_all():
    query = model_query(models.ComputeHostReservation, get_session())
    return query.all()


def _host_reservation_get_by_reservation_id(session, reservation_id):
    query = model_query(models.ComputeHostReservation, session)
    return query.filter_by(reservation_id=reservation_id).first()


def host_reservation_get_by_reservation_id(reservation_id):
    return _host_reservation_get_by_reservation_id(get_session(),
                                                   reservation_id)


def host_reservation_create(values):
    values = values.copy()
    host_reservation = models.ComputeHostReservation()
    host_reservation.update(values)

    session = get_session()
    with session.begin():
        try:
            host_reservation.save(session=session)
        except common_db_exc.DBDuplicateEntry as e:
            # raise exception about duplicated columns (e.columns)
            raise db_exc.ClimateDBDuplicateEntry(
                model=host_reservation.__class__.__name__, columns=e.columns)

    return host_reservation_get(host_reservation.id)


def host_reservation_update(host_reservation_id, values):
    session = get_session()

    with session.begin():
        host_reservation = _host_reservation_get(session,
                                                 host_reservation_id)
        host_reservation.update(values)
        host_reservation.save(session=session)

    return host_reservation_get(host_reservation_id)


def host_reservation_destroy(host_reservation_id):
    session = get_session()
    with session.begin():
        host_reservation = _host_reservation_get(session,
                                                 host_reservation_id)

        if not host_reservation:
            # raise not found error
            raise db_exc.ClimateDBNotFound(
                id=host_reservation_id, model='ComputeHostReservation')

        session.delete(host_reservation)


# ComputeHostAllocation
def _host_allocation_get(session, host_allocation_id):
    query = model_query(models.ComputeHostAllocation, session)
    return query.filter_by(id=host_allocation_id).first()


def host_allocation_get(host_allocation_id):
    return _host_allocation_get(get_session(),
                                host_allocation_id)


def host_allocation_get_all():
    query = model_query(models.ComputeHostAllocation, get_session())
    return query.all()


def host_allocation_get_all_by_values(**kwargs):
    """Returns all entries filtered by col=value."""
    allocation_query = model_query(models.ComputeHostAllocation, get_session())
    for name, value in kwargs.items():
        column = getattr(models.ComputeHostAllocation, name, None)
        if column:
            allocation_query = allocation_query.filter(column == value)
    return allocation_query.all()


def host_allocation_create(values):
    values = values.copy()
    host_allocation = models.ComputeHostAllocation()
    host_allocation.update(values)

    session = get_session()
    with session.begin():
        try:
            host_allocation.save(session=session)
        except common_db_exc.DBDuplicateEntry as e:
            # raise exception about duplicated columns (e.columns)
            raise db_exc.ClimateDBDuplicateEntry(
                model=host_allocation.__class__.__name__, columns=e.columns)

    return host_allocation_get(host_allocation.id)


def host_allocation_update(host_allocation_id, values):
    session = get_session()

    with session.begin():
        host_allocation = _host_allocation_get(session,
                                               host_allocation_id)
        host_allocation.update(values)
        host_allocation.save(session=session)

    return host_allocation_get(host_allocation_id)


def host_allocation_destroy(host_allocation_id):
    session = get_session()
    with session.begin():
        host_allocation = _host_allocation_get(session,
                                               host_allocation_id)

        if not host_allocation:
            # raise not found error
            raise db_exc.ClimateDBNotFound(
                id=host_allocation_id, model='ComputeHostAllocation')

        session.delete(host_allocation)


def host_allocation_optimize():
#    plugin = host_plugin.PhysicalHostPlugin()
    # Create a reservation list R
    leases = lease_get_all()
    leases = [lease for lease in leases if lease['start_date'] > datetime.datetime.now()]
    # Sort R by length
    leases = sorted(leases, key=lambda k: k['end_date'] - k['start_date'], reverse=True)
    for lease in leases:
        reservations = reservation_get_all_by_lease_id(lease['id'])
        # For each r in R:
        for reservation in reservations:
            host_reservations = host_reservation_get_by_reservation_id(reservation['id'])
            if not isinstance(host_reservations, (list)):
                host_reservations = [host_reservations]
            for host_reservation in host_reservations:
                # Create a list of matching hosts H
                hosts = _matching_hosts(host_reservation['hypervisor_properties'], host_reservation['resource_properties'],
                                        host_reservation['count_range'], lease['start_date'], lease['end_date'], hardware_only=True)
                # Joindre la liste des hotes alloues actuellement
                for alloc in host_allocation_get_all_by_values(reservation_id=reservation['id']):
                    if alloc['compute_host_id'] not in hosts:
                        hosts.append(alloc['compute_host_id'])
                # Sort this list by decreasing efficiency
                ks = client.Client(auth_url='http://10.5.5.5:35357/v2.0', username='kwranking', password='password', tenant_name='service')
                endpoint = ks.service_catalog.url_for(service_type='efficiency', endpoint_type='publicURL')
                kwr = kwrclient.Client('1', endpoint, ks.auth_token)
                hosts_lst = ''
                for x in hosts:
                    hosts_lst += x + ';'
                hosts = kwr.node.rank_hosts_list({'hosts': hosts_lst, 'method': 'Efficiency', 'number': len(hosts)})
                # Create a list of current allocations
                current_allocations = host_allocation_get_all_by_values(reservation_id=reservation['id'])
                # Sort this list by increasing efficiency
                current_hosts = [h['compute_host_id'] for h in current_allocations]
                hosts_lst = ''
                for x in current_hosts:
                    hosts_lst += x + ';'
                current_hosts = kwr.node.rank_hosts_list({'hosts': hosts_lst, 'method': 'Efficiency', 'number': len(current_hosts)})
                current_hosts = current_hosts[::-1]
                current_allocations_sorted = []
                for h in current_hosts:
                    current_allocations_sorted += [alloc for alloc in current_allocations if alloc['compute_host_id'] == h]
                if hosts[:len(current_hosts)] == current_hosts[::-1]:
                    break
                # Put r on H (if ok), try the other ones otherwise
                for worst_allocation in current_allocations_sorted:
                    for host in hosts:
                        if worst_allocation['compute_host_id'] == host:
                            break
                        allocation = host_allocation_get_all_by_values(
                                compute_host_id=host)
                        if not allocation:
                            host_allocation_destroy(worst_allocation['id'])
                            host_allocation_create({'compute_host_id': host,
                                              'reservation_id': reservation['id']})
                            break
                        elif db_utils.get_free_periods(
                            host,
                            lease['start_date'],
                            lease['end_date'],
                            lease['end_date'] - lease['start_date'],
                        ) == [
                            (lease['start_date'], lease['end_date']),
                        ]:
                            host_allocation_destroy(worst_allocation['id'])
                            host_allocation_create({'compute_host_id': host,
                                              'reservation_id': reservation['id']})
                            break


def _matching_hosts(hypervisor_properties, resource_properties,
                    count_range, start_date, end_date, hardware_only=False):
    """Return the matching hosts (preferably not allocated)

    """
    count_range = count_range.split('-')
    min_host = count_range[0]
    max_host = count_range[1]
    allocated_host_ids = []
    not_allocated_host_ids = []
    filter_array = []
    # TODO(frossigneux) support "or" operator
    if hypervisor_properties:
        filter_array = _convert_requirements(
            hypervisor_properties)
    if resource_properties:
        filter_array += _convert_requirements(
            resource_properties)
    return [host['id'] for host in host_get_all_by_queries(filter_array)]


def _convert_requirements(requirements):
    """Convert the requirements to an array of strings.
    ["key op value", "key op value", ...]

    """
    # TODO(frossigneux) Support the "or" operator
    # Convert text to json
    if isinstance(requirements, six.string_types):
        try:
            requirements = json.loads(requirements)
        except ValueError:
            raise manager_ex.MalformedRequirements(rqrms=requirements)

    # Requirement list looks like ['<', '$ram', '1024']
    if _requirements_with_three_elements(requirements):
        result = []
        if requirements[0] == '=':
            requirements[0] = '=='
        string = (requirements[1][1:] + " " + requirements[0] + " " +
                  requirements[2])
        result.append(string)
        return result
    # Remove the 'and' element at the head of the requirement list
    elif _requirements_with_and_keyword(requirements):
        return [_convert_requirements(x)[0]
                for x in requirements[1:]]
    # Empty requirement list0
    elif isinstance(requirements, list) and not requirements:
        return requirements
    else:
        raise manager_ex.MalformedRequirements(rqrms=requirements)


def _requirements_with_three_elements(requirements):
    """Return true if requirement list looks like ['<', '$ram', '1024']."""
    return (isinstance(requirements, list) and
            len(requirements) == 3 and
            isinstance(requirements[0], six.string_types) and
            isinstance(requirements[1], six.string_types) and
            isinstance(requirements[2], six.string_types) and
            requirements[0] in ['==', '=', '!=', '>=', '<=', '>', '<'] and
            len(requirements[1]) > 1 and requirements[1][0] == '$' and
            len(requirements[2]) > 0)


#ComputeHost
def _host_get(session, host_id):
    query = model_query(models.ComputeHost, session)
    return query.filter_by(id=host_id).first()


def _host_get_all(session):
    query = model_query(models.ComputeHost, session)
    return query


def host_get(host_id):
    return _host_get(get_session(), host_id)


def host_list():
    return model_query(models.ComputeHost, get_session()).all()


def host_get_all_by_filters(filters):
    """Returns hosts filtered by name of the field."""

    hosts_query = _host_get_all(get_session())

    if 'status' in filters:
        hosts_query = hosts_query.filter(
            models.ComputeHost.status == filters['status'])

    return hosts_query.all()


def host_get_all_by_queries(queries):
    """Returns hosts filtered by an array of queries.

    :param queries: array of queries "key op value" where op can be
        http://docs.sqlalchemy.org/en/rel_0_7/core/expression_api.html
            #sqlalchemy.sql.operators.ColumnOperators

    """
    hosts_query = model_query(models.ComputeHost, get_session())

    oper = {
        '<': ['lt', lambda a, b: a >= b],
        '>': ['gt', lambda a, b: a <= b],
        '<=': ['le', lambda a, b: a > b],
        '>=': ['ge', lambda a, b: a < b],
        '==': ['eq', lambda a, b: a != b],
        '!=': ['ne', lambda a, b: a == b],
    }

    hosts = []
    for query in queries:
        try:
            key, op, value = query.split(' ', 3)
        except ValueError:
            raise db_exc.ClimateDBInvalidFilter(query_filter=query)

        column = getattr(models.ComputeHost, key, None)
        if column:
            if op == 'in':
                filt = column.in_(value.split(','))
            else:
                if op in oper:
                    op = oper[op][0]
                try:
                    attr = filter(lambda e: hasattr(column, e % op),
                                  ['%s', '%s_', '__%s__'])[0] % op
                except IndexError:
                    raise db_exc.ClimateDBInvalidFilterOperator(
                        filter_operator=op)

                if value == 'null':
                    value = None

                filt = getattr(column, attr)(value)

            hosts_query = hosts_query.filter(filt)
        else:
            # looking for extra capabilities matches
            extra_filter = model_query(
                models.ComputeHostExtraCapability, get_session()
            ).filter(models.ComputeHostExtraCapability.capability_name == key
                     ).all()
            if not extra_filter:
                raise db_exc.ClimateDBNotFound(
                    id=key, model='ComputeHostExtraCapability')

            for host in extra_filter:
                if op in oper and oper[op][1](host.capability_value, value):
                    hosts.append(host.computehost_id)
                elif op not in oper:
                    msg = 'Operator %s for extra capabilities not implemented'
                    raise NotImplementedError(msg % op)

    return hosts_query.filter(~models.ComputeHost.id.in_(hosts)).all()


def host_create(values):
    values = values.copy()
    host = models.ComputeHost()
    host.update(values)

    session = get_session()
    with session.begin():
        try:
            host.save(session=session)
        except common_db_exc.DBDuplicateEntry as e:
            # raise exception about duplicated columns (e.columns)
            raise db_exc.ClimateDBDuplicateEntry(
                model=host.__class__.__name__, columns=e.columns)

    return host_get(host.id)


def host_update(host_id, values):
    session = get_session()

    with session.begin():
        host = _host_get(session, host_id)
        host.update(values)
        host.save(session=session)

    return host_get(host_id)


def host_destroy(host_id):
    session = get_session()
    with session.begin():
        host = _host_get(session, host_id)

        if not host:
            # raise not found error
            raise db_exc.ClimateDBNotFound(id=host_id, model='Host')

        session.delete(host)


# ComputeHostExtraCapability
def _host_extra_capability_get(session, host_extra_capability_id):
    query = model_query(models.ComputeHostExtraCapability, session)
    return query.filter_by(id=host_extra_capability_id).first()


def host_extra_capability_get(host_extra_capability_id):
    return _host_extra_capability_get(get_session(),
                                      host_extra_capability_id)


def _host_extra_capability_get_all_per_host(session, host_id):
    query = model_query(models.ComputeHostExtraCapability, session)
    return query.filter_by(computehost_id=host_id)


def host_extra_capability_get_all_per_host(host_id):
    return _host_extra_capability_get_all_per_host(get_session(),
                                                   host_id).all()


def host_extra_capability_create(values):
    values = values.copy()
    host_extra_capability = models.ComputeHostExtraCapability()
    host_extra_capability.update(values)

    session = get_session()
    with session.begin():
        try:
            host_extra_capability.save(session=session)
        except common_db_exc.DBDuplicateEntry as e:
            # raise exception about duplicated columns (e.columns)
            raise db_exc.ClimateDBDuplicateEntry(
                model=host_extra_capability.__class__.__name__,
                columns=e.columns)

    return host_extra_capability_get(host_extra_capability.id)


def host_extra_capability_update(host_extra_capability_id, values):
    session = get_session()

    with session.begin():
        host_extra_capability = (
            _host_extra_capability_get(session,
                                       host_extra_capability_id))
        host_extra_capability.update(values)
        host_extra_capability.save(session=session)

    return host_extra_capability_get(host_extra_capability_id)


def host_extra_capability_destroy(host_extra_capability_id):
    session = get_session()
    with session.begin():
        host_extra_capability = (
            _host_extra_capability_get(session,
                                       host_extra_capability_id))

        if not host_extra_capability:
            # raise not found error
            raise db_exc.ClimateDBNotFound(
                id=host_extra_capability_id,
                model='ComputeHostExtraCapability')

        session.delete(host_extra_capability)


def host_extra_capability_get_all_per_name(host_id, capability_name):
    session = get_session()

    with session.begin():
        query = _host_extra_capability_get_all_per_host(session, host_id)
        return query.filter_by(capability_name=capability_name).all()
