# NimbusDesk Webhooks

NimbusDesk webhooks can trigger on ticket created, ticket updated, ticket closed,
customer created, and satisfaction rating submitted events.

Webhook payloads include the event id, workspace id, timestamp, actor id, and
changed fields. Failed webhook deliveries are retried for 12 hours.

Each workspace can define 25 active webhooks on the Team plan and 100 active
webhooks on the Scale plan.

Metadata: product=nimbusdesk, doc_type=api, plan=all
