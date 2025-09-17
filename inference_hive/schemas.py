"""
OpenAI Python SDK Compatible PyArrow Schemas.
Compatible with v1.86.0. Subject to change.
See https://github.com/openai/openai-python for the latest version.
"""
import pyarrow as pa


COMPLETION_SCHEMA = pa.schema([
    pa.field('id', pa.string()),
    pa.field('response', pa.struct([
        pa.field('id', pa.string()),
        pa.field('choices', pa.list_(
            pa.struct([
                pa.field('finish_reason', pa.string()),
                pa.field('index', pa.int32()),
                pa.field('logprobs', pa.struct([
                    pa.field('text_offset', pa.list_(pa.int32())),
                    pa.field('token_logprobs', pa.list_(pa.float32())),
                    pa.field('tokens', pa.list_(pa.string())),
                    pa.field('top_logprobs', pa.list_(pa.map_(pa.string(), pa.float32()))),
                ])),
                pa.field('text', pa.string()),
            ])
        )),
        pa.field('created', pa.int64()),
        pa.field('model', pa.string()),
        pa.field('system_fingerprint', pa.string()),
        pa.field('usage', pa.struct([
            pa.field('completion_tokens', pa.int32()),
            pa.field('prompt_tokens', pa.int32()),
            pa.field('total_tokens', pa.int32()),
            pa.field('completion_tokens_details', pa.struct([
                pa.field('accepted_prediction_tokens', pa.int32()),
                pa.field('audio_tokens', pa.int32()),
                pa.field('reasoning_tokens', pa.int32()),
                pa.field('rejected_prediction_tokens', pa.int32()),
            ])),
            pa.field('prompt_tokens_details', pa.struct([
                pa.field('audio_tokens', pa.int32()),
                pa.field('cached_tokens', pa.int32()),
            ])),
        ])),
    ]))
])

CHAT_COMPLETION_SCHEMA = pa.schema([
    pa.field('id', pa.string()),
    pa.field('response', pa.struct([
        pa.field('id', pa.string()),
        pa.field('choices', pa.list_(
            pa.struct([
                pa.field('finish_reason', pa.string()),
                pa.field('index', pa.int32()),
                pa.field('logprobs', pa.list_(pa.struct([
                    pa.field('content', pa.struct([
                        pa.field('token', pa.string()),
                        pa.field('bytes', pa.list_(pa.int32())),
                        pa.field('logprob', pa.float32()),
                        pa.field('top_logprobs', pa.list_(pa.struct([
                            pa.field('token', pa.string()),
                            pa.field('bytes', pa.list_(pa.int32())),
                            pa.field('logprob', pa.float32()),
                        ]))),
                    ])),
                    pa.field('refusal', pa.struct([
                        pa.field('token', pa.string()),
                        pa.field('bytes', pa.list_(pa.int32())),
                        pa.field('logprob', pa.float32()),
                        pa.field('top_logprobs', pa.list_(pa.struct([
                            pa.field('token', pa.string()),
                            pa.field('bytes', pa.list_(pa.int32())),
                            pa.field('logprob', pa.float32()),
                        ]))),
                    ])),
                ]))),
                pa.field('message', pa.struct([
                    pa.field('content', pa.string()),
                    pa.field('refusal', pa.string()),
                    pa.field('role', pa.string()),
                    pa.field('annotations', pa.list_(pa.struct([
                        pa.field('type', pa.string()),
                        pa.field('url_citation', pa.struct([
                            pa.field('end_index', pa.int32()),
                            pa.field('start_index', pa.int32()),
                            pa.field('title', pa.string()),
                            pa.field('url', pa.string()),
                        ])),
                    ]))),
                    pa.field('audio', pa.struct([
                        pa.field('id', pa.string()),
                        pa.field('data', pa.string()),
                        pa.field('expires_at', pa.int64()),
                        pa.field('transcript', pa.string()),
                    ])),
                    pa.field('function_call', pa.struct([
                        pa.field('arguments', pa.string()),
                        pa.field('name', pa.string()),
                    ])),
                    pa.field('tool_calls', pa.list_(pa.struct([
                        pa.field('id', pa.string()),
                        pa.field('function', pa.struct([
                            pa.field('arguments', pa.string()),
                            pa.field('name', pa.string()),
                        ])),
                        pa.field('type', pa.string()),
                    ])))
                ])),
            ])),
        ),
        pa.field('created', pa.int64()),
        pa.field('model', pa.string()),
        pa.field('object', pa.string()),
        pa.field('service_tier', pa.string()),
        pa.field('system_fingerprint', pa.string()),
        pa.field('usage', pa.struct([
            pa.field('completion_tokens', pa.int32()),
            pa.field('prompt_tokens', pa.int32()),
            pa.field('total_tokens', pa.int32()),
            pa.field('completion_tokens_details', pa.struct([
                pa.field('accepted_prediction_tokens', pa.int32()),
                pa.field('audio_tokens', pa.int32()),
                pa.field('reasoning_tokens', pa.int32()),
                pa.field('rejected_prediction_tokens', pa.int32()),
            ])),
            pa.field('prompt_tokens_details', pa.struct([
                pa.field('audio_tokens', pa.int32()),
                pa.field('cached_tokens', pa.int32()),
            ])),
        ])),
    ]))
])
